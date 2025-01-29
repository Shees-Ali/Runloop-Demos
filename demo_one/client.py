import sys
import os
from dotenv import load_dotenv
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
import logging
import openai

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATABASE_FILE = "Chinook_Sqlite.sqlite"
RUNLOOP_API_KEY_ENV = "RUNLOOP_API_KEY"
DEVBOX_ID_ENV = "DEVBOX_ID"
OPENAI_API_KEY_ENV = "OPENAI_API"

def generate_sql_query(prompt: str, schema_info: str) -> str:
        """
        Generate SQL query using OpenAI API based on natural language prompt and schema information.
        
        Args:
            prompt (str): Natural language query prompt
            schema_info (str): Database schema information to provide context
            
        Returns:
            str: Generated SQL query
        """
        try:
            system_prompt = f"""
            You are an SQL expert. Generate a SQL query based on the user's request.
            Use the following schema information to create an accurate query:
            {schema_info}
            
            Return ONLY the SQL query, nothing else.
            """
            
            openai.api_key = os.getenv(OPENAI_API_KEY_ENV)
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Lower temperature for more focused SQL generation
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate query: {e}")
            raise

def analyze_data_structure(query_response: str) -> str:
    """
    Analyzes the structure of query response data as a JSON string.
    
    Args:
        query_response (str): Query result as a JSON string
        
    Returns:
        str: Detailed description of data structure and suitable visualization approaches
    """
    try:
        import json
        
        # Parse JSON string
        data = json.loads(query_response)
        if not data:
            return "Empty dataset"
            
        # Get sample record
        sample_record = data[0]
        
        # Analyze structure
        analysis = []
        analysis.append(f"Dataset contains {len(data)} records")
        
        # Categorize fields
        numeric_fields = []
        date_fields = []
        categorical_fields = []
        
        for field, value in sample_record.items():
            if isinstance(value, (int, float)):
                numeric_fields.append(field)
            elif isinstance(value, str):
                # Check if it might be a date
                if 'date' in field.lower() or 'time' in field.lower():
                    date_fields.append(field)
                else:
                    categorical_fields.append(field)
                    
        # Add field information
        if numeric_fields:
            analysis.append(f"Numeric fields: {', '.join(numeric_fields)}")
        if date_fields:
            analysis.append(f"Date/Time fields: {', '.join(date_fields)}")
        if categorical_fields:
            analysis.append(f"Categorical fields: {', '.join(categorical_fields)}")
            
        # Suggest visualizations
        suggestions = []
        if date_fields and numeric_fields:
            suggestions.append("Time series plots for temporal analysis")
        if len(numeric_fields) >= 2:
            suggestions.append("Scatter plots or line charts for numeric relationships")
        if categorical_fields and numeric_fields:
            suggestions.append("Bar charts or box plots for categorical-numeric relationships")
        if categorical_fields:
            suggestions.append("Pie charts or bar charts for categorical distributions")
            
        if suggestions:
            analysis.append("Suggested visualizations: " + "; ".join(suggestions))
            
        return "\n".join(analysis)
    except Exception as e:
        logger.error(f"Error analyzing data structure: {e}")
        return f"Error analyzing data structure: {str(e)}"

def generate_visualization_code(query_response: str, prompt_description: str) -> str:
    """
    Generate Python code for visualizing JSON string data using OpenAI API.

    Args:
        query_response (str): Query result as a JSON string
        prompt_description (str): Description of how the data should be visualized

    Returns:
        str: Python code for visualization
    """
    try:
        data_analysis = analyze_data_structure(query_response)
        
        openai.api_key = os.getenv(OPENAI_API_KEY_ENV)

        system_prompt = f"""
        You are an expert data visualization assistant. 
        Generate the raw Python code to visualize JSON string data using pandas and matplotlib.

        Data Structure Analysis:
        {data_analysis}

        Data Sample (as JSON string):
        {query_response}

        Requirements:
        - Convert the JSON string to DataFrame using pd.read_json with StringIO
        - Handle datetime fields appropriately using pd.to_datetime
        - Use matplotlib for visualizations
        - Include proper labels, titles, and legends
        - Match the visualization to the user's description
        - Ensure the code is well-commented
        - Handle potential errors (empty data, missing values, etc.)
        - Use plt.tight_layout() for better spacing
        - Set appropriate figure size using plt.figure(figsize=(width, height))

        VERY IMPORTANT: 
        - Do NOT include any markdown code block markers (```python or ```)
        - Do NOT include any explanatory text
        - Return ONLY the raw Python code itself
        - The code should start directly with Python statements
        - The code should end with the last Python statement
        """

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_description}
            ],
            temperature=0.1
        )

        # Get the response and clean it
        code = response.choices[0].message.content.strip()
        
        print("\n \n Code:", code)
        
        # Remove any potential markdown code blocks
        if code.startswith('```python'):
            code = code[9:]
        elif code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
            
        return code.strip()

    except Exception as e:
        logger.error(f"Error generating visualization code: {e}")
        raise

def visualize_query_results(query_response: str, prompt_description: str):
    """
    Generate and execute Python code to visualize query results from JSON string data.

    Args:
        query_response (str): Query results as a JSON string
        prompt_description (str): Description of how the data should be visualized
    """
    try:
        import json
        from io import StringIO
        
        # First analyze and print data structure
        print("\nAnalyzing data structure...")
        data_analysis = analyze_data_structure(query_response)
        print(data_analysis)
        
        # Generate and print visualization code
        print("\nGenerating visualization code based on data analysis...")
        visualization_code = generate_visualization_code(query_response, prompt_description)
        print("\nGenerated Visualization Code:")
        print(visualization_code)

        # Create DataFrame from JSON string
        df = pd.read_json(StringIO(query_response))
        
        # Convert date columns to datetime
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])

        # Create a restricted environment for code execution
        restricted_globals = {
            "pd": pd,
            "plt": plt,
            "df": df,
            "np": np,
            "json": json,
            "StringIO": StringIO
        }
        
        # Execute the visualization code
        print("\nExecuting visualization...")
        exec(visualization_code, restricted_globals, {})
        plt.show()

    except Exception as e:
        print(f"Error executing visualization: {e}")
        print("Detailed error:", str(e))
        
async def interact_with_mcp_server(db_path: str):
    """
    Interacts with the SQLite MCP server to perform database operations using natural language queries.
    
    Args:
        db_path (str): Path to the SQLite database file.
    """
    server_params = StdioServerParameters(
        command="python",
        args=["server.py", db_path],
        env=None
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # Get schema information for context
                tables_result = await session.call_tool("list_tables")
                tables = eval(tables_result.content[0].text)
                
                # Build schema information for OpenAI
                schema_info = []
                for table in tables:
                    table_name = table['name']
                    schema_result = await session.call_tool(
                        "describe_table", 
                        arguments={"table_name": table_name}
                    )
                    schema_info.append(f"Table {table_name}:")
                    schema_info.append(schema_result.content[0].text)
                
                schema_context = "\n".join(schema_info)
                
                while True:
                    # Get natural language query from user
                    print("\nEnter your query in natural language (or 'exit' to quit):")
                    user_input = input()
                    
                    if user_input.lower() == 'exit':
                        break
                    
                    try:
                        # Generate SQL query using OpenAI
                        print("\nGenerating SQL query...")
                        sql_query = generate_sql_query(user_input, schema_context)
                        print(f"\nGenerated SQL Query:\n{sql_query}")
                        
                        # Execute the generated query
                        print("\nExecuting query...")
                        result = await session.call_tool(
                            "read_query",
                            arguments={"query": sql_query}
                        )
                        
                        # Print results
                        print("\nQuery Results:")
                        print(result.content[0].text)
                        
                        result = result.content[0].text
                        
                        # Add insight
                        insight = f"Query executed: {user_input}"
                        await session.call_tool(
                            "append_insight",
                            arguments={"insight": insight}
                        )
                        
                        # restart the loop
                        continue
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        print("Please try rephrasing your query.")
                        
                print("\nEnter your query for visualization natural language (or 'exit' to quit):")
                user_input = input()
                
                if user_input.lower() == 'exit':
                    # end the session
                    return

                # Generate Code for Visualization using OpenAI
                print("\nGenerating Visualization Code...")
                visualize_query_results(result, user_input)

    except Exception as e:
        print(f"Error during server interaction: {e}")
    except asyncio.CancelledError:
        print("Operation was canceled")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <db_path>")
        sys.exit(1)

    db_path = sys.argv[1]  # Get the database path from command-line arguments
    import asyncio
    asyncio.run(interact_with_mcp_server(db_path))