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
        "The system must guarantee breakfast delivery in less than 25 minutes to all parts of the city.",
        "The system must allow offering prepackaged breakfasts like mini-breakfast, luxury breakfast, etc.",
        "Customers must be able to assemble individual breakfasts from simple products.",
        "Prepackaged products may contain simple products or other prepackaged products.",
        "Typical orders must consist of various amounts of prepackaged and/or simple products.",
        "Each product must have associated attributes: unit (e.g., grams) and price (in Euros).",
        "Customers must be able to place orders only over the phone.",
        "Customers must call the company number and provide their customer number.",
        "The system must validate customer numbers based on their format (area code, digits, checksum).",
        "The system must not allow collective orders from several customers.",
        "The system must authenticate customers, including a check for blacklisted customers.",
        "Customers must be able to add products directly to the shopping cart by naming them.",
        "Customers must be able to request suitable product recommendations based on specified criteria (e.g., calorie count and price).",
        "Customers must be able to use a previous order as a blueprint for a new order.",
        "The system must allow combining multiple methods of assembling a shopping cart within a single order but restrict each order to one blueprint at most.",
        "An order must be able to serve as a blueprint many times.",
        "The system must store one predefined address per customer for delivery.",
        "Packing clerks must be able to assemble orders based on shopping cart contents.",
        "Packing clerks must be able to print a label for the order containing the packing clerk's name, customer's first name, surname, address, order number, and assigned delivery clerk.",
        "Packing clerks must be able to attach the labels to the paper bags.",
        "Packing clerks must be able to print an invoice showing the label data plus the ordered products, their quantities, and total price.",
        "Packing clerks must ensure each reprinted invoice has a unique copy number.",
        "Delivery clerks must calculate optimal itineraries for delivering multiple orders.",
        "Delivery clerks must print itineraries, take the corresponding bags and invoices, and collect customer signatures for order confirmation.",
        "Delivery clerks must ensure customers sign a copy of the invoice, which is retained by the company, while the customer keeps another copy.",
        "Customers must be able to inquire about the status of an order by providing the order number.",
        "The system must indicate whether the delivery clerk is on the way.",
        "Customers must be able to cancel an order by providing the order number, but only before the order has been assembled.",
        "Canceled orders cannot be undone.",
        "Orders cannot be updated; customers must cancel and place a new order if changes are needed.",
        "The system must be web-based, automating all current processes, replacing phone ordering.",
        "The system must replace text processing systems for labeling.",
        "The system must replace spreadsheet tools for itinerary calculation.",
        "The application must support the following browsers (specific versions to be defined).",
        "The user groups are customers, packing clerks, delivery clerks, and managers.",
        "Customers must confirm deliveries via browser on a smartphone by entering a password.",
        "The application must provide a browser-based, unauthenticated product search feature.",
        "Customers must be able to place orders via text message using a predefined string format including customer number, password, product codes, and quantities.",
        "Each product must have a unique product code for SMS orders.",
        "The system must handle SMS orders within the limitations of text message length.",
        "The system must respond to SMS orders with a text message containing the assigned order number.",
        "Customers must be able to cancel orders via text message by providing the order number in a predefined format.",
        "The system must generate and transfer payment records to the existing payment system upon order assembly.",
        "Payment records must include customer number, order number, total amount in Euros, and expected payment date.",
        "The system must automatically generate and print a nightly business report for managers, detailing orders of the day, products, amounts, packing clerks, delivery clerks, customers, addresses, and order numbers."
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
        embeddings.append(response.data[0].embedding)

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
    output = {f"Cluster {i}": [requirements[j] for j, label in enumerate(clusters) if label == i] for i in
              range(num_clusters)}
    with open("output/clusters.json", "w") as f:
        json.dump(output, f, indent=2)


def visualize_embeddings(embeddings, clusters):
    print("Visualizing clusters...")

    # Convert embeddings to a NumPy array
    embeddings_array = np.array(embeddings)

    # Check shape to ensure it's suitable for t-SNE
    if len(embeddings_array.shape) != 2:
        raise ValueError(f"Expected embeddings to be a 2D array, got shape {embeddings_array.shape}")

    # Ensure clusters match the number of embeddings
    if len(embeddings_array) != len(clusters):
        raise ValueError(f"Mismatch: embeddings have {len(embeddings_array)} elements but clusters have {len(clusters)}")

    # Set perplexity dynamically
    n_samples = embeddings_array.shape[0]
    perplexity = min(30, n_samples - 1)  # Perplexity must be less than n_samples

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(embeddings_array)

    # Define distinct colors for clusters
    cluster_ids = np.unique(clusters)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'lime', 'pink', 'brown', 'gray']
    if len(cluster_ids) > len(colors):
        raise ValueError("Not enough colors defined for the number of clusters.")
    color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(cluster_ids)}
    cluster_colors = [color_map[cluster] for cluster in clusters]

    # Scatter plot of reduced embeddings
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=cluster_colors
    )

    # Create a legend with cluster labels and move it outside the plot
    legend_labels = [f"Cluster {cluster_id}" for cluster_id in cluster_ids]
    handles = [plt.Line2D([0], [0], marker='o', color=color_map[cluster_id], markersize=10, linestyle='') for cluster_id
               in cluster_ids]
    plt.legend(
        handles, legend_labels, title="Clusters", loc="upper left",
        bbox_to_anchor=(1.05, 1), borderaxespad=0
    )

    # Add title and description
    plt.title("Requirement Clusters (t-SNE Visualization)")
    plt.xlabel("Dimension 1 (Reduced)")
    plt.ylabel("Dimension 2 (Reduced)")

    # Add a description below the diagram
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

    upsert_embeddings_to_pinecone(pinecone_index, embeddings)

    query_vector = embeddings[0]  # Use the first embedding as the query vector
    result = pinecone_index.query(vector=query_vector, top_k=len(embeddings), include_values=True)
    data = [res['values'] for res in result['matches']]

    num_clusters = 12
    clusters = cluster_embeddings(embeddings, requirements, num_clusters)
    save_clusters_to_file(clusters, requirements, num_clusters)

    visualize_embeddings(data, clusters)


# Execute the main function
if __name__ == "__main__":
    main()
