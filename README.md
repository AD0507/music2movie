# **Cross-Media Recommendation System: Songs to Movies**

This project builds a **cross-media recommendation system** that suggests movies based on the "vibe" of a selected song. It leverages **deep learning models** and machine learning techniques to combine audio features, textual data, and movie poster embeddings for personalized recommendations.

---

# **Dataset Used**

1. Spotify Tracks Dataset. Hugging Face. Retrieved from-  https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
2. TMDb Movies Dataset 2023 (930K+ Movies). Kaggle. Retrieved from-  https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies

---

## **Project Structure**

The project repository includes the following components:

### **1. Streamlit App (Demo)**
- The **Streamlit app** is built on a **subset of the data** for demonstration purposes.
- The actual dataset is large, and generating real-time recommendations for the full dataset in Streamlit would be computationally intensive.
- The code for this subset-based app can be found under the **`code`** folder:
  - **`Final Code-Using Samples.ipynb`**: Notebook for the demo app built on a smaller dataset.

### **2. Main Code (Full Dataset Recommendations)**
- The complete implementation, which generates recommendations for the **entire dataset**, is available in:
  - **`Recommendation system code.ipynb`** (located under the `code` folder).
- This notebook includes the full pipeline for embedding generation, feature extraction, and recommendation outputs.

### **3. Exploratory Data Analysis (EDA)**
- A separate notebook for **Exploratory Data Analysis (EDA)** is provided to give insights into the dataset before building the recommendation system:
  - **`EDA.ipynb`** (located under the `code` folder).

---

## **How Deep Learning Was Used**

Deep learning played a significant role in the project for feature extraction from movie poster images:

1. **Poster Embedding Generation using MobileNetV2**:
   - A pre-trained **MobileNetV2** model (trained on ImageNet) was used to extract **visual features** from movie poster images.
   - Steps:
     - Posters were resized and preprocessed to match MobileNetV2’s input requirements.
     - The images were passed through MobileNetV2’s convolutional layers (excluding the classification layer) to generate **1280-dimensional embeddings**.
   - These embeddings capture the visual "vibe" of the movies, such as colors, textures, and patterns.

2. **Combining Modalities**:
   - Poster embeddings were combined with:
     - **Textual Features**: Sentiment scores from movie overviews.
     - **Numerical Features**: Normalized runtime, popularity, and revenue.
   - This created a **multimodal movie embedding**, which was then compared to song embeddings.

---

## **Project Workflow**

1. **Data Preparation**:
   - Movie and Spotify datasets were cleaned and processed.
   - EDA was performed to understand key patterns in the data.

2. **Feature Extraction**:
   - **Movies**:
     - Visual embeddings from posters (MobileNetV2).
     - Sentiment scores from overviews.
     - Normalized numerical features.
   - **Songs**:
     - Normalized audio features like tempo, danceability, and energy.

3. **Similarity Calculation**:
   - Song embeddings were compared with movie embeddings using **cosine similarity**.

4. **Recommendation Output**:
   - Movies were ranked based on similarity scores, and the top recommendations were presented.

---

## **How to Use the Project**

1. **Streamlit App**:
   - For demonstration purposes, run the Streamlit app using the subset-based code:
     ```bash
     streamlit run Final Code-Using Samples.ipynb
     ```

2. **Full Recommendations**:
   - To explore recommendations for the entire dataset, execute the **`Recommendation system code.ipynb`** notebook.

3. **EDA**:
   - Analyze the data and visualizations using **`EDA.ipynb`**.

---

## **Key Features**
- Cross-media recommendation system connecting movies and songs.
- Use of **deep learning (MobileNetV2)** for extracting poster embeddings.
- Integration of multimodal features (image, text, and numerical).
- Streamlit app for easy demo and visualization.

---

## **Technologies Used**
- **Deep Learning**: TensorFlow, MobileNetV2
- **NLP**: TextBlob (sentiment analysis)
- **Machine Learning**: Cosine similarity, feature normalization
- **Libraries**: Pandas, NumPy, Scikit-learn, Streamlit, SentenceTransformers (optional)

---

## **Acknowledgements**
This project uses pre-trained deep learning models and data from public sources, including:
- TMDB movie dataset
- Spotify audio features dataset
- MobileNetV2 (TensorFlow)

---
