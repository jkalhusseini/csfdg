# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
from PIL import Image

# Change the theme to "dark"
#st.theme("light")

# Enable wide mode to occupy the whole width of the page
st.set_page_config(layout="wide")

# Disable the warning related to Matplotlib's global figure object
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define CHLA logo
chla_logo = Image.open('chla_logo.png')

# Display CHLA logo in the top left corner
st.image(chla_logo, use_column_width=False, width=350)

# Page title and description
st.title("CHLA CSF Flow Dynamics Data Analysis App")
st.write("Upload an Excel file with data for analysis.")
st.caption("This app was built by Jacob K. Al-Husseini & Joseph H. Ha")

# Upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Read the Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)

    # Remove the "Patient No." and "MRN" columns if they exist
    if "Patient No." in df.columns:
        df.drop(columns=["Patient No."], inplace=True)
    if "MRN" in df.columns:
        df.drop(columns=["MRN"], inplace=True)

    # Convert the "Date of Initial Shunt Placement" column to datetime
    if "Date of Initial Shunt Placement" in df.columns:
        df["Date of Initial Shunt Placement"] = pd.to_datetime(df["Date of Initial Shunt Placement"])

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.write(df)

    # Data Analysis
    st.subheader("Data Analysis:")

    # Select columns for analysis
    selected_columns = st.multiselect("Select columns for analysis:", df.columns)

    if selected_columns:
        # Scatter plot using Plotly
        st.subheader("Scatter Plot:")
        x_axis = st.selectbox("Select X-axis:", selected_columns)
        y_axis = st.selectbox("Select Y-axis:", selected_columns)
        scatter_fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(scatter_fig)

        # Histograms using Seaborn
        st.subheader("Histograms:")
        num_cols = len(selected_columns)
        num_cols_per_row = 3  # Number of histograms per row
        num_rows = num_cols // num_cols_per_row + (num_cols % num_cols_per_row > 0)

        for i in range(num_rows):
            cols_in_this_row = selected_columns[i * num_cols_per_row: (i + 1) * num_cols_per_row]
            row = st.columns(num_cols_per_row)
            for j, col in enumerate(cols_in_this_row):
                if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                    hist_fig, ax = plt.subplots()
                    sns.histplot(data=df, x=col, ax=ax)
                    row[j].pyplot(hist_fig)

        # Box plots using Seaborn
        st.subheader("Box Plots:")
        num_cols = len(selected_columns)
        num_cols_per_row = 3  # Number of box plots per row
        num_rows = num_cols // num_cols_per_row + (num_cols % num_cols_per_row > 0)

        for i in range(num_rows):
            cols_in_this_row = selected_columns[i * num_cols_per_row: (i + 1) * num_cols_per_row]
            row = st.columns(num_cols_per_row)
            for j, col in enumerate(cols_in_this_row):
                if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                    box_fig, ax = plt.subplots()
                    sns.boxplot(data=df, y=col, ax=ax)
                    row[j].pyplot(box_fig)

        # Categorical Data Analysis using Seaborn
        st.subheader("Categorical Data Analysis:")
        for column in selected_columns:
            if df[column].dtype == 'object':
                st.write(f"**{column} Distribution:**")
                value_counts = df[column].value_counts()
                bar_fig, ax = plt.subplots()
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                st.pyplot(bar_fig)

        # Multivariate Analysis - Correlation Heatmap
        st.subheader("Multivariate Analysis - Correlation Heatmap:")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) > 1:
            corr = df[numeric_columns].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
            st.pyplot()

        # Multivariate Analysis - Pairplot
        st.subheader("Multivariate Analysis - Pairplot:")
        if len(numeric_columns) > 1:
            pairplot_fig = sns.pairplot(df[numeric_columns])
            st.pyplot(pairplot_fig)

        # Multivariate Analysis - Cluster Analysis
        st.subheader("Multivariate Analysis - Cluster Analysis:")
        num_clusters = st.slider("Select the number of clusters:", min_value=2, max_value=10)
        if len(numeric_columns) > 1:
            # Impute missing values with column means
            imputer = SimpleImputer(strategy='mean')
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

            kmeans = KMeans(n_clusters=num_clusters)
            df['Cluster'] = kmeans.fit_predict(df[numeric_columns])
            cluster_fig = px.scatter(df, x=x_axis, y=y_axis, color='Cluster', title=f"Cluster Analysis (K={num_clusters})")
            st.plotly_chart(cluster_fig)

        # Multivariate Analysis - PCA
        st.subheader("Multivariate Analysis - PCA:")
        if len(numeric_columns) > 1:
            # Standardize the data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_columns])

            # Apply PCA
            pca = PCA(n_components=2)  # You can choose the number of components
            pca_result = pca.fit_transform(scaled_data)

            # Create a DataFrame with PCA results
            pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])

            # Plot PCA results
            pca_fig = px.scatter(pca_df, x='PCA1', y='PCA2', title="PCA Analysis")
            st.plotly_chart(pca_fig)

        # 3D Surface Plot
        st.subheader("3D Surface Plot:")
        if len(numeric_columns) >= 3:
            x_surface = st.selectbox("Select X-axis for Surface Plot:", numeric_columns)
            y_surface = st.selectbox("Select Y-axis for Surface Plot:", numeric_columns)
            z_surface = st.selectbox("Select Z-axis for Surface Plot:", numeric_columns)
    
        if len(set([x_surface, y_surface, z_surface])) == 3:
            surface_fig = go.Figure(data=[go.Surface(z=df[z_surface], x=df[x_surface], y=df[y_surface])])
            surface_fig.update_layout(scene=dict(zaxis_title=z_surface, xaxis_title=x_surface, yaxis_title=y_surface))
        
        # Add scatter points to make the plot more informative
        scatter_trace = go.Scatter3d(
            x=df[x_surface],
            y=df[y_surface],
            z=df[z_surface],
            mode='markers',
            marker=dict(size=3, opacity=0.7, color='red'),
            name='Data Points'
        )
        surface_fig.add_trace(scatter_trace)

        st.plotly_chart(surface_fig)


        # Classification Algorithm Selection
        st.subheader("Classification Analysis:")

        # Select target and features columns
        target_column = st.selectbox("Select the target column:", df.columns)
        feature_columns = st.multiselect("Select feature columns:", df.columns)

        # Split data into train and test sets
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Classification Algorithm Selection
        st.subheader("Select a Classification Algorithm:")
        classifier_name = st.selectbox("Select an algorithm:", ["Random Forest", "Logistic Regression"])

        if classifier_name == "Random Forest":
            classifier = RandomForestClassifier(random_state=42)
        elif classifier_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression()

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test)

        # Display classification metrics
        st.subheader("Classification Metrics:")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

        st.write("Classification Report:")
        classification_rep = classification_report(y_test, y_pred)
        st.text(classification_rep)

        st.write("Confusion Matrix:")
        confusion_mat = confusion_matrix(y_test, y_pred)
        st.write(confusion_mat)

        # Allow users to input data for predictions
        st.subheader("Make Predictions:")
        input_data = {}
        for column in feature_columns:
            input_data[column] = st.text_input(f"Enter {column}:")
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = classifier.predict(input_df)
            st.write(f"Predicted {target_column}: {prediction[0]}")

else:
    st.write("Please upload an Excel file.")

# Footer
st.write("Built with Streamlit")
