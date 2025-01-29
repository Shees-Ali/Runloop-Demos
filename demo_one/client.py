import sys
import os
from dotenv import load_dotenv
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
import logging
import openai

import pandas as pd
import matplotlib.pyplot as plt
import ast


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

def generate_visualization_code(query_response: str, prompt_description: str) -> str:
    """
    Use OpenAI API to generate Python code for visualizing database query results.

    Args:
        query_response (str): Query result as a string containing list of dictionaries.
        prompt_description (str): A natural language description of how the data should be visualized.

    Returns:
        str: Python code for visualization.
    """
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        system_prompt = f"""
        You are an expert data visualization assistant. 
        Generate Python code to visualize the given data using pandas and matplotlib.

        Data example:
        {query_response}

        Requirements:
        - First convert the string data to DataFrame using ast.literal_eval
        - Use matplotlib for visualizations
        - Include proper labels and titles
        - Match the visualization to the user's description

        Only return the Python code.
        """

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_description}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error generating visualization code: {e}")
        raise

def visualize_query_results(query_response: str, prompt_description: str):
    """
    Generate and execute Python code to visualize query results dynamically.

    Args:
        query_response (str): Query results as a string containing list of dictionaries.
        prompt_description (str): Description of how the data should be visualized.

    Returns:
        None
    """
    try:
        visualization_code = generate_visualization_code(query_response, prompt_description)

        print("\nGenerated Visualization Code:")
        print(visualization_code)

        # Convert string to list of dictionaries and create DataFrame
        data = ast.literal_eval(query_response)
        df = pd.DataFrame(data)

        restricted_globals = {
            "__builtins__": {"print": print, "pd": pd, "plt": plt},
            "df": df
        }
        exec(visualization_code, restricted_globals, {})
        plt.show()

    except Exception as e:
        print(f"Error executing visualization: {e}")

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