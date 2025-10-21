from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Handle imports for both module and standalone execution
try:
    from .base_agent import BaseAgent
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base_agent import BaseAgent

class QueryGeneratorAgent(BaseAgent):
    """
    Agent responsible for converting subqueries into optimized Google search prompts
    """
    
    def __init__(self):
        super().__init__(
            name="QueryGeneratorAgent",
            description="Converts subqueries into optimized Google search prompts"
        )
        self.prompt_template = self._create_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for generating 5 professional, detailed Google queries per subquery."""
        template = """
        Transform this research subquery into exactly 5 Google search queries.

        Subquery: {subquery}

        REQUIREMENTS:
        - Use professional search operators (site:, filetype:, intitle:, AND, OR)
        - Include authoritative sources (scholar.google.com, bloomberg.com, mckinsey.com, etc.)
        - Add time modifiers (2024, recent, latest)
        - Each query should be 8-15 words

        RESPOND WITH ONLY A JSON ARRAY - NO OTHER TEXT:
        ["query1", "query2", "query3", "query4", "query5"]
        """

        return PromptTemplate(
            input_variables=["subquery"],
            template=template
        )
    
    def _save_results(self, results: Dict[str, Any], original_query: str = None) -> None:
        """Save the query generation results in both JSON and Markdown formats."""
        try:
            # Create timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Determine output directory (relative to the script location)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "..", "output", "google_query")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save JSON format
            json_filename = f"google_queries_{timestamp}.json"
            json_path = os.path.join(output_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save Markdown format
            md_filename = f"google_queries_{timestamp}.md"
            md_path = os.path.join(output_dir, md_filename)
            
            markdown_content = self._format_markdown(results, original_query)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            self.log(f"Results saved to: {json_path} and {md_path}")
            
        except Exception as e:
            self.log(f"Error saving results: {str(e)}")
    
    def _format_markdown(self, results: Dict[str, Any], original_query: str = None) -> str:
        """Format the query generation results as a professional Markdown report."""
        data = results.get("data", {})
        status = results.get("status", "unknown")
        
        # Header section
        markdown = "# Google Search Query Generation Report\n\n"
        markdown += f"**Generated:** {data.get('timestamp', 'Unknown')}\n"
        markdown += f"**Status:** {status.upper()}\n"
        markdown += f"**Agent:** QueryGeneratorAgent\n"
        
        if original_query:
            markdown += f"**Original Research Query:** {original_query}\n"
        
        markdown += f"**Subqueries Processed:** {data.get('subquery_count', 0)}\n\n"
        
        # Error handling
        if status == "error":
            markdown += "## ⚠️ Error Information\n\n"
            markdown += f"**Error:** {data.get('error', 'Unknown error')}\n\n"
        
        # Results section
        markdown += "## 🔍 Generated Google Search Queries\n\n"
        
        generated_queries = data.get("generated", [])
        
        if not generated_queries:
            markdown += "*No queries were generated.*\n\n"
        else:
            for idx, item in enumerate(generated_queries, 1):
                subquery = item.get("original_subquery", "Unknown")
                queries = item.get("queries", [])
                
                markdown += f"### {idx}. Research Subquery\n\n"
                markdown += f"**Original Subquery:** `{subquery}`\n\n"
                markdown += "**Professional Google Search Queries:**\n\n"
                
                for q_idx, query in enumerate(queries, 1):
                    # Escape any markdown special characters in the query
                    escaped_query = query.replace('`', '\\`').replace('*', '\\*').replace('_', '\\_')
                    markdown += f"{q_idx}. `{escaped_query}`\n"
                
                markdown += "\n"
        
        # Footer section
        markdown += "---\n\n"
        markdown += "## 📊 Summary Statistics\n\n"
        markdown += f"- **Total Subqueries:** {len(generated_queries)}\n"
        markdown += f"- **Total Queries Generated:** {sum(len(item.get('queries', [])) for item in generated_queries)}\n"
        markdown += f"- **Average Queries per Subquery:** {sum(len(item.get('queries', [])) for item in generated_queries) / max(len(generated_queries), 1):.1f}\n\n"
        
        markdown += "## 🎯 Search Strategy Notes\n\n"
        markdown += "- Queries are optimized for professional research and analysis\n"
        markdown += "- Advanced search operators are used for precision and authority\n"
        markdown += "- Time-based modifiers ensure current and relevant results\n"
        markdown += "- Site-specific operators target authoritative sources\n"
        markdown += "- Boolean logic enhances search comprehensiveness\n\n"
        
        markdown += "*Generated by QueryGeneratorAgent - Professional Search Query Optimization System*\n"
        
        return markdown
 


    async def execute(self, input_data: List[str]) -> Dict[str, Any]:
        """
        Generate 5 professional Google queries for each subquery.

        Args:
            input_data: List of subqueries to convert (can be strings or structured objects)

        Returns:
            Dict with per-subquery arrays of 5 optimized professional queries
        """
        try:
            self.log(f"Generating professional search queries for {len(input_data)} subqueries")
            results = []
            
            for i, subquery_item in enumerate(input_data):
                # Handle both old format (strings) and new format (structured objects)
                if isinstance(subquery_item, dict):
                    subquery_text = subquery_item.get("text", "")
                    subquery_id = subquery_item.get("id", f"subquery_{i}")
                else:
                    subquery_text = str(subquery_item)
                    subquery_id = f"subquery_{i}"
                
                if not subquery_text.strip():
                    continue
                
                self.log(f"Processing subquery {i+1}/{len(input_data)}: {subquery_id}")
                
                # Generate queries for this specific subquery
                chain_result = await self.chain.arun(subquery=subquery_text)
                
                # Parse the LLM output
                try:
                    queries = json.loads(chain_result.strip())
                    if not isinstance(queries, list):
                        raise ValueError("Result is not a list")
                except (json.JSONDecodeError, ValueError):
                    # Simple fallback parsing
                    lines = chain_result.strip().split('\n')
                    queries = [line.strip().strip('"').strip("'") for line in lines if line.strip()]
                    queries = [q for q in queries if q and not q.startswith('[') and not q.startswith(']')]
                
                # Take only the first 5 queries if more are generated
                queries = queries[:5]
                
                # Create result object
                result_item = {
                    "subquery_id": subquery_id,
                    "original_subquery": subquery_text,
                    "queries": queries,
                    "query_count": len(queries)
                }
                
                results.append(result_item)
                self.log(f"Generated {len(queries)} queries for subquery: {subquery_id}")
            
            # Create output data
            output_data = {
                "subquery_count": len(results),
                "total_queries_generated": sum(len(item["queries"]) for item in results),
                "generated": results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results automatically
            self._save_results({"data": output_data, "status": "success"})
            
            return self.format_output(output_data)
            
        except Exception as e:
            self.log(f"Error generating queries: {str(e)}")
            error_data = {
                "error": str(e),
                "subquery_count": len(input_data) if input_data else 0,
                "generated": [],
                "timestamp": datetime.now().isoformat()
            }
            return self.format_output(error_data, status="error")


# Runnable entry point to process a planner JSON and generate queries
async def main():
    """Main function to run the QueryGeneratorAgent"""
    
    # Automatically find the latest planner output file
    planner_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "planneroutput")
    
    if not os.path.exists(planner_output_dir):
        print(f"❌ Planner output directory not found: {planner_output_dir}")
        return
    
    # Find the latest JSON file
    json_files = [f for f in os.listdir(planner_output_dir) if f.endswith('.json')]
    if not json_files:
        print(f"❌ No JSON files found in: {planner_output_dir}")
        return
    
    # Get the most recent file
    latest_file = max(json_files, key=lambda f: os.path.getctime(os.path.join(planner_output_dir, f)))
    planner_json_path = os.path.join(planner_output_dir, latest_file)
    
    print(f"📁 Loading planner output from: {latest_file}")
    
    try:
        with open(planner_json_path, "r", encoding="utf-8") as f:
            planner_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read JSON: {e}")
        return

    # Extract subqueries from the planner output
    subqueries = planner_data.get("subqueries")
    if not isinstance(subqueries, list) or not subqueries:
        print("❌ No valid 'subqueries' array found in planner JSON")
        return

    # Get original query for context
    original_query = planner_data.get("original_query", "Unknown")
    
    print(f"🎯 Processing {len(subqueries)} subqueries from planner output...")
    print(f"📝 Original Query: {original_query[:100]}{'...' if len(original_query) > 100 else ''}")
    print()
    
    # Initialize the agent and run execution
    agent = QueryGeneratorAgent()
    result = await agent.execute(subqueries)
    
    # Print results to console in a readable format
    status = result.get("status", "success")
    data = result.get("data", {})
    
    print(f"\n{'='*60}")
    print("🎯 PROFESSIONAL GOOGLE SEARCH QUERIES GENERATED")
    print(f"{'='*60}")
    print(f"Original Query: {original_query}")
    print(f"Status: {status.upper()}")
    print(f"Subqueries processed: {data.get('subquery_count', 0)}")
    print(f"Total queries generated: {data.get('total_queries_generated', 0)}")
    print()

    for item in data.get("generated", []):
        subquery_id = item.get('subquery_id', 'Unknown')
        original_subquery = item.get('original_subquery', 'Unknown')
        queries = item.get('queries', [])
        
        print(f"📋 Subquery ID: {subquery_id}")
        print(f"   Text: {original_subquery}")
        print("🔍 Professional Google Search Queries:")
        for i, q in enumerate(queries, 1):
            print(f"   {i}. {q}")
        print()
    
    print("💾 Results automatically saved to backend/output/google_query/")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())