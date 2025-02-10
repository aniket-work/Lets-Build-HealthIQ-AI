# pages/02_vector_space.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA

from core.embeddings import EmbeddingsManager
from src.constants import SETTINGS_PATH, CONFIG_PATH
from src.utils import load_yaml_config, load_json_config, setup_environment
from src.session_manager import initialize_components, get_embeddings_data


def create_vector_plot(embeddings_3d, hover_texts, colors, sizes, search_3d=None):
    """Create 3D vector space plot."""
    data = [
        go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Documents'
        )
    ]

    if search_3d is not None:
        data.append(
            go.Scatter3d(
                x=search_3d[:, 0],
                y=search_3d[:, 1],
                z=search_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='diamond',
                    color='rgba(0,255,0,0.8)',
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                name='Search Query'
            )
        )

    fig = go.Figure(data=data)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3',
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True if search_3d is not None else False,
        height=700
    )

    return fig


def main():
    st.set_page_config(layout="wide")

    try:
        # Load configurations
        settings = load_yaml_config(SETTINGS_PATH)
        config = load_json_config(CONFIG_PATH)
        setup_environment(config["api_keys"]["huggingface"])

        # Check if embeddings already exist in session state
        if "embeddings_data" not in st.session_state or st.session_state.embeddings_data is None:
            # Only initialize components if we don't have embeddings data
            _, embeddings_manager, _, _, _ = initialize_components(settings, config)
        else:
            # Just initialize embeddings_manager for search functionality
            embeddings_manager = EmbeddingsManager(
                model_name=settings["model"]["embeddings"]["name"]
            )

        # Get embeddings data from session state
        data = get_embeddings_data()
        if data is None:
            st.error("No embeddings data found. Please return to the main page and try again.")
            return

        # Create sidebar
        st.sidebar.title("Search Controls")
        search_query = st.sidebar.text_input("Search in documents")

        # Main content
        st.title("Document Vector Space")

        col1, col2 = st.columns([7, 3])

        with col1:
            if data["embeddings"] is not None:
                # Convert embeddings to numpy array if needed
                embeddings = np.array(data["embeddings"])

                # Perform PCA
                pca = PCA(n_components=3)
                embeddings_3d = pca.fit_transform(embeddings)

                # Create hover texts
                hover_texts = [
                    f"Document {i}\nSource: {m.get('source', 'unknown')}\nContent: {doc[:200]}..."
                    for i, (m, doc) in enumerate(zip(data["metadata"], data["documents"]))
                ]

                # Initialize colors and sizes
                colors = ['rgba(100,100,255,0.6)'] * len(embeddings_3d)
                sizes = [6] * len(embeddings_3d)

                search_3d = None
                if search_query:
                    # Get query embedding and similar vectors
                    query_vector = embeddings_manager.get_query_embedding(search_query)
                    similar_indices = embeddings_manager.find_similar_vectors(
                        query_vector, embeddings, k=5
                    )

                    # Highlight similar vectors
                    for idx in similar_indices:
                        colors[idx] = 'rgba(255,50,50,0.8)'
                        sizes[idx] = 10

                    # Transform query vector
                    search_3d = pca.transform(query_vector)

                # Create and display plot
                fig = create_vector_plot(embeddings_3d, hover_texts, colors, sizes, search_3d)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if search_query:
                # Display similar documents
                query_vector = embeddings_manager.get_query_embedding(search_query)
                similar_indices = embeddings_manager.find_similar_vectors(
                    query_vector, embeddings, k=5
                )

                st.subheader("Similar Documents")
                for idx in similar_indices:
                    with st.expander(f"Document {idx}"):
                        st.write(data["documents"][idx][:200] + "...")
            else:
                st.info("Enter a search term to find similar documents")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Try refreshing the page if initialization fails")


if __name__ == "__main__":
    main()