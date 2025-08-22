import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

def generate_visualization(df, column_name=None, column_names=None):
    """
    Generate visualization based on column types
    Returns: base64 encoded image or None if no visualization can be generated
    """
    try:
        plt.figure(figsize=(10, 6))
        
        if column_names and len(column_names) == 2:
            # Two columns visualization
            col1, col2 = column_names
            
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                # Scatter plot for two numeric columns
                plt.scatter(df[col1], df[col2])
                plt.title(f'Relationship between {col1} and {col2}')
                plt.xlabel(col1)
                plt.ylabel(col2)
            elif pd.api.types.is_numeric_dtype(df[col1]) and not pd.api.types.is_numeric_dtype(df[col2]):
                # Box plot for numeric vs categorical
                sns.boxplot(x=col2, y=col1, data=df)
                plt.title(f'{col1} by {col2}')
                plt.xticks(rotation=45)
            elif not pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                # Box plot for categorical vs numeric
                sns.boxplot(x=col1, y=col2, data=df)
                plt.title(f'{col2} by {col1}')
                plt.xticks(rotation=45)
            else:
                # Count plot for two categorical columns
                pd.crosstab(df[col1], df[col2]).plot(kind='bar')
                plt.title(f'Relationship between {col1} and {col2}')
                plt.xlabel(col1)
                plt.xticks(rotation=45)
                plt.legend(title=col2)
                
        elif column_name:
            # Single column visualization
            if pd.api.types.is_numeric_dtype(df[column_name]):
                # Histogram for numeric columns
                plt.hist(df[column_name].dropna(), bins=20, edgecolor='black')
                plt.title(f'Distribution of {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Frequency')
            else:
                # Bar chart for categorical columns
                df[column_name].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
        else:
            return None
            
        plt.tight_layout()
        
        # Convert plot to base64 encoded image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        return plot_url
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        plt.close()
        return None

def enhanced_chat_csv(df, query, llm):
    """
    Enhanced CSV chat that can generate visualizations
    Returns: (answer, visualization_image) tuple
    """
    try:
        # Create a prompt that asks Gemini to suggest visualizations
        prompt_template = f"""
        You are an expert data analyst. Here is the CSV data with columns: {list(df.columns)}.
        First 5 rows:
        {df.head().to_string()}

        Based on this data, answer the question: {query}
        
        Additionally, if appropriate, suggest a visualization that would help understand the data.
        Format your response as:
        ANSWER: [your concise answer]
        VISUALIZE: [column_name] or [column1,column2] or None
        
        If a visualization would be helpful, specify the column name(s) after VISUALIZE:.
        If no visualization is needed, just write VISUALIZE: None.
        """

        response = llm.invoke(prompt_template)
        response_text = response.content
        
        # Parse the response
        answer = "No answer found"
        visualize_cols = None
        
        for line in response_text.split('\n'):
            if line.startswith('ANSWER:'):
                answer = line.replace('ANSWER:', '').strip()
            elif line.startswith('VISUALIZE:'):
                cols = line.replace('VISUALIZE:', '').strip()
                if cols.lower() != 'none':
                    visualize_cols = [c.strip() for c in cols.split(',')]
        
        # Generate visualization if suggested
        visualization_img = None
        if visualize_cols:
            if len(visualize_cols) == 1:
                visualization_img = generate_visualization(df, column_name=visualize_cols[0])
            elif len(visualize_cols) == 2:
                visualization_img = generate_visualization(df, column_names=visualize_cols)
        
        return answer, visualization_img
        
    except Exception as e:
        return f"Error: {str(e)}", None