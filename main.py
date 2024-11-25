import os
import json
import numpy as np
from dotenv import load_dotenv
import openai
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Load environment variables
load_dotenv()


# Initialize OpenAI API
def initialize_openai():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in the .env file.")
    return OpenAI(api_key=openai_api_key)


# Requirements list
def get_requirements():
    return [
        "Deliver breakfast to all parts of the city within 25 minutes.",
        "Provide prepackaged breakfasts (e.g., mini-breakfast, luxury breakfast).",
        "Allow customers to assemble individual breakfasts by choosing from a product list.",
        "Support prepackaged products that may contain other prepackaged and/or simple products.",
        "Process orders with various products in different amounts (prepackaged and simple products).",
        "Maintain unit-based pricing denominated in Euros.",
        "Authenticate customers by their customer number.",
        "Block blacklisted customers from placing orders.",
        "Allow customers to name a previous order as a blueprint for a new order.",
        "Permit order customization through: Naming products.Choosing from a product list.Reusing a previous order.",
        "Restrict orders to one delivery address per customer.",
        "Provide an order number after the order is placed.",
        "Allow packing clerks to assemble orders manually.",
        "Generate a label with customer and delivery details for each order.",
        "Print invoices showing detailed order information.",
        "Ensure that multiple invoice copies are numbered separately.",
        "Optimize delivery routes for delivery clerks.",
        "Provide order status inquiries over the phone.",
        "Allow order cancellations (only before assembly is complete).",
        "Restrict updates to canceling and replacing orders.",
        "Automate ordering, labeling, and route calculation with a web-based application.",
        "Replace phone-based order placement with a web interface.",
        "Allow customers to search for products without authentication.",
        "Introduce SMS-based order placement with specific formats for placing and canceling orders.",
        "Integrate with the existing payment system by transferring payment records after packing.",
        "Enable browser-based delivery confirmations via customer password entry.",
        "Automatically generate a daily business report listing order details (products, quantities, clerks, customers, "
        "addresses, etc.)."
    ]


# Generate or load embeddings
def get_embeddings(oai, requirements):
    embeddings_file = 'output/embeddings.txt'
    if os.path.exists(embeddings_file):
        print("Embeddings file found. Loading embeddings from file.")
        with open(embeddings_file, 'r') as f:
            return [json.loads(line.strip()) for line in f.readlines()]

    print("Generating embeddings...")
    embeddings = []
    for req in requirements:
        response = oai.embeddings.create(
            model="text-embedding-ada-002",
            input=req
        )
        embeddings.append(response['data'][0]['embedding'])

    # Save embeddings to a file
    with open(embeddings_file, 'w') as f:
        for emb in embeddings:
            f.write(f'{json.dumps(emb)}\n')

    return embeddings


# Initialize Pinecone and return the index
def setup_pinecone():
    print("Setting up Pinecone...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not found. Please set PINECONE_API_KEY in the .env file.")

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "earlybird-requirements"

    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pc.Index(index_name)


# Upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings):
    print("Upserting embeddings into Pinecone index...")
    for i, embedding in enumerate(embeddings):
        index.upsert([(f"requirement-{i}", embedding)])


# Cluster embeddings using KMeans
def cluster_embeddings(embeddings, requirements, num_clusters=5):
    print("Clustering embeddings...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    for cluster_id in range(num_clusters):
        print(f"Cluster {cluster_id}:")
        for idx, label in enumerate(clusters):
            if label == cluster_id:
                print(f"  - {requirements[idx]}")

    return clusters


# Save clusters to a file
def save_clusters_to_file(clusters, requirements, num_clusters):
    print("Saving clusters to file...")
    output = {f"Cluster {i}": [requirements[j] for j, label in enumerate(clusters) if label == i] for i in range(num_clusters)}
    with open("output/clusters.json", "w") as f:
        json.dump(output, f, indent=2)


def visualize_embeddings(embeddings, clusters):
    print("Visualizing clusters...")

    # Convert embeddings to a NumPy array
    embeddings_array = np.array(embeddings)

    # Check shape to ensure it's suitable for t-SNE
    if len(embeddings_array.shape) != 2:
        raise ValueError(f"Expected embeddings to be a 2D array, got shape {embeddings_array.shape}")

    # Set perplexity dynamically
    n_samples = embeddings_array.shape[0]
    perplexity = min(30, n_samples - 1)  # Perplexity must be less than n_samples

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(embeddings_array)

    # Scatter plot of reduced embeddings
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=clusters, cmap="viridis"
    )
    plt.colorbar(scatter, label="Cluster ID")

    # Add title and description
    plt.title("Requirement Clusters (t-SNE Visualization)")
    plt.xlabel("Dimension 1 (Reduced)")
    plt.ylabel("Dimension 2 (Reduced)")

    # Add a legend describing the purpose of the diagram
    plt.figtext(
        0.5, -0.05,
        "This diagram visualizes the relative similarity of requirements, reduced from high-dimensional embeddings into 2D space. Points with similar meanings are grouped together.",
        wrap=True, horizontalalignment='center', fontsize=10
    )

    plt.tight_layout()
    plt.savefig('output/clusters.png')  # Save the plot as an image
    plt.show()


# Main function to orchestrate the steps
def main():
    oai = initialize_openai()
    requirements = get_requirements()

    embeddings = get_embeddings(oai, requirements)
    pinecone_index = setup_pinecone()

    # upsert_embeddings_to_pinecone(pinecone_index, embeddings)

    query_vector = embeddings[0]  # Use the first embedding as the query vector
    result = pinecone_index.query(vector=query_vector, top_k=len(embeddings), include_values=True)
    data = [res['values'] for res in result['matches']]

    num_clusters = 5
    clusters = cluster_embeddings(embeddings, requirements, num_clusters)
    save_clusters_to_file(clusters, requirements, num_clusters)

    visualize_embeddings(data, clusters)


# Execute the main function
if __name__ == "__main__":
    main()
