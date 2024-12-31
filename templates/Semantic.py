import streamlit as st
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from sentence_transformers import SentenceTransformer, util
import requests
import numpy as np

# ----------------------------
# IBM Watson Discovery credentials
COLLECTION_ID = 'a6dfd26d-cf20-9123-0000-0193539248de'
DISCOVERY_API_KEY = "cTB8m6bfNkrx9lzYVub3COxeWvpmpfQgcezenWxZRGaG"
DISCOVERY_URL = "https://api.us-south.discovery.watson.cloud.ibm.com/instances/1e0a56cc-4091-4367-aa3e-1b4b2f7640e5"
PROJECT_ID = "4f51758b-abac-4b28-8204-9a8590ce2b91"
VERSION = "2021-11-30"

# Watsonx.ai credentials
WATSONX_API_KEY = 'CNpSGAiXJ_u4fdeyTIeADtH2NM6HOrg5bt8N4XKx-_02'
WATSONX_URL = 'https://eu-gb.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29'
WATSONX_MODEL_ID = 'meta-llama/llama-3-1-70b-instruct'
WATSONX_PROJECT_ID = '4dddc81b-96a4-4455-aea3-40e9b1a48ffa'

# Sentence Transformer model for embeddings
MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME)

# ----------------------------
# Initialize IBM Watson Discovery
# ----------------------------
def initialize_discovery(api_key, url):
    authenticator = IAMAuthenticator(api_key)
    discovery = DiscoveryV2(version=VERSION, authenticator=authenticator)
    discovery.set_service_url(url)
    return discovery

discovery = initialize_discovery(DISCOVERY_API_KEY, DISCOVERY_URL)

# ----------------------------
# Fetch Specific Document by Document ID
# ----------------------------
def fetch_document_by_id(discovery, project_id, collection_id, document_id):
    try:
        response = discovery.query(
            project_id=project_id,
            collection_ids=[collection_id],
            filter=f"document_id::{document_id}",
            return_=["text"]
        ).get_result()
        results = response.get("results", [])
        if results:
            # Return the document text as a string
            return results[0].get("text", "")
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching document: {e}")
        return None

# ----------------------------
# Semantic Search Function
# ----------------------------
def semantic_search(query, document_text, top_k=3):
    # Ensure document_text is a string
    if isinstance(document_text, list):
        document_text = "\n\n".join(document_text)  # Join list elements into a single string

    # Split document into sections (e.g., paragraphs)
    sections = document_text.split("\n\n")

    # Generate embeddings for the sections
    section_embeddings = embedding_model.encode(sections, convert_to_tensor=True).cpu().numpy()

    # Compute the embedding for the query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy()

    # Perform cosine similarity search
    scores = util.cos_sim(query_embedding, section_embeddings)[0].cpu().numpy()
    indices = np.argsort(scores)[::-1][:top_k]  # Top-k highest scores

    # Retrieve matching sections
    results = [{"text": sections[idx], "score": scores[idx]} for idx in indices]
    return results

# ----------------------------
# Watsonx.ai Response Generation
# ----------------------------
def get_watsonx_response(query, context):
    # Generate Bearer token
    token_response = requests.post(
        url="https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=f"apikey={WATSONX_API_KEY}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    )
    if token_response.status_code != 200:
        return None, "Failed to generate Watsonx.ai token"
    bearer_token = token_response.json().get('access_token')

    # Combine query and context
    combined_input = f"{query}\n\nContext:\n{context}"

    # Call Watsonx.ai API
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {bearer_token}"}
    payload = {
        "input": combined_input,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 300,
            "min_new_tokens": 0,
            "stop_sequences": [],
            "repetition_penalty": 1
        },
        "model_id": WATSONX_MODEL_ID,
        "project_id": WATSONX_PROJECT_ID
    }
    response = requests.post(WATSONX_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get('results', [{}])[0].get('generated_text', "No response generated."), None
    else:
        return None, f"Error calling Watsonx.ai: {response.status_code}"

# ----------------------------
# Streamlit Application
# ----------------------------
def main():
    st.title("Crisp Responses with Watson Discovery and Watsonx.ai")

    # Input fields for Document ID and Query
    document_id = st.text_input("Enter Document ID:")
    query = st.text_input("Enter your query:")

    if st.button("Fetch and Generate"):
        if document_id.strip():
            # Fetch specific document by its ID
            st.write(f"Fetching details for Document ID: {document_id}...")
            document_text = fetch_document_by_id(discovery, PROJECT_ID, COLLECTION_ID, document_id)

            if document_text:
                #st.write("Document retrieved successfully!")
                #st.write(f"**Document Content (Snippet):** {document_text[:300]}...")

                # Perform semantic search within the document
                results = semantic_search(query, document_text)
                st.subheader("Semantic Search Results")
                context = ""
                for result in results:
                    st.write(f"- **Relevant Section:** {result['text'][:300]}... (Score: {result['score']:.4f})")
                    context += result['text'] + "\n\n"

                # Generate a concise response using Watsonx.ai
                st.subheader("Watsonx.ai Response")
                response, error = get_watsonx_response(query, context)
                if error:
                    st.error(error)
                else:
                    st.markdown(response)
            else:
                st.warning("No document found for the specified ID.")
        else:
            st.warning("Please enter a valid Document ID.")

if __name__ == "__main__":
    main()
