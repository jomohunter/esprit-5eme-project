# Standard typing imports for type hints used throughout this module
from typing import Dict, Any, List

# LangChain prompt and chain utilities to build an LLM workflow
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Built-in modules for parsing, filesystem operations, and runtime configuration
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Handle imports so the file can run both as a package module and standalone script
try:
    # Preferred relative import when using this file inside the 'agents' package
    from .base_agent import BaseAgent
except ImportError:
    # If running directly via 'python planner_agent.py', adjust sys.path to find base_agent
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Fallback absolute import after path adjustment
    from base_agent import BaseAgent

class PlannerAgent(BaseAgent):
    """
    High-level planning agent:
    - Understands a user's research query
    - Produces comprehensive subqueries to guide downstream research agents
    """
    
    def __init__(self):
        # Initialize the base class with a name and description for logging/metadata
        super().__init__(
            name="PlannerAgent",
            description="Analyzes user queries and breaks them into actionable subqueries"
        )
        # Build the prompt template that guides the LLM to produce subqueries
        self.prompt_template = self._create_prompt_template()
        # Create an LLMChain that couples the LLM with the prompt, so we can run it easily
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create and return the prompt template used to instruct the LLM."""
        # The prompt template instructs the LLM to produce 5 professional subqueries
        # and return them in a strict JSON array format. {query} is injected at runtime.
        template = """
        You are an expert research strategist and information analyst. Your mission is to thoroughly understand the user's research query and transform it into a comprehensive research plan with detailed, professional subqueries.

        User Query: {query}

        Your task is to:
        1. Analyze the query deeply to understand the user's information needs
        2. Generate exactly 5 comprehensive, professional subqueries
        3. Each subquery should be a complete, detailed research question (3-5 lines each)
        4. Cover different aspects: current state, recent developments, key players, challenges, future outlook
        5. Make each subquery specific, actionable, and suitable for thorough web research

        Format your response as a JSON array with exactly 5 detailed subqueries. Each subquery should be comprehensive and professional.

        Example for "What are the latest developments in renewable energy technology?":
        [
            "What are the most significant technological breakthroughs and innovations in renewable energy systems (solar, wind, hydro, geothermal) that have emerged in 2024, including efficiency improvements, cost reductions, and new materials or designs that are revolutionizing the industry?",
            "Which major corporations, startups, and research institutions are leading the renewable energy technology development, what are their latest projects and investments, and how are they positioning themselves in the competitive landscape of clean energy solutions?",
            "What are the current market trends, adoption rates, and economic factors driving renewable energy technology deployment globally, including government policies, subsidies, and regulatory frameworks that are accelerating or hindering progress?",
            "What are the primary technical challenges, limitations, and barriers facing renewable energy technologies today, such as energy storage, grid integration, intermittency issues, and infrastructure requirements that need to be addressed for widespread adoption?",
            "What are the projected future developments, emerging technologies, and long-term outlook for renewable energy systems over the next 5-10 years, including potential game-changing innovations and their expected impact on global energy transition?"
        ]

        Now generate 5 comprehensive, professional subqueries for the user's query. Return only the JSON array:
        """

        # Return a PromptTemplate that binds the above template and its variables
        return PromptTemplate(
            input_variables=["query"],  # The template expects a single variable called 'query'
            template=template            # The instruction text shown to the LLM
        )
    
    async def execute(self, input_data: str) -> Dict[str, Any]:
        """
        Orchestrate planning for a given user query.
        - Runs the LLM chain with the prompt
        - Parses the output into a list of subqueries
        - Creates structured subquery objects with IDs and metadata
        - Persists results and returns a standardized response dict
        """
        try:
            # Log the start of the planning process with the input query
            self.log(f"Planning research for query: {input_data}")

            # Run the prompt-driven chain asynchronously to get the raw LLM output
            result = await self.chain.arun(query=input_data)

            # Attempt to parse the raw output as JSON (expected: a list of strings)
            try:
                subqueries_raw = json.loads(result.strip())
                # Validate the parsed structure to ensure it's a list
                if not isinstance(subqueries_raw, list):
                    raise ValueError("Result is not a list")
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, fall back to splitting lines and cleaning text
                lines = result.strip().split('\n')
                # Strip quotes and whitespace, then filter out bracket-only lines
                subqueries_raw = [line.strip().strip('"').strip("'") for line in lines if line.strip()]
                subqueries_raw = [q for q in subqueries_raw if q and not q.startswith('[') and not q.startswith(']')]

            # Create structured subquery objects with IDs and metadata
            structured_subqueries = []
            current_timestamp = datetime.now().isoformat()
            
            for i, subquery_text in enumerate(subqueries_raw, 1):
                # Generate a unique ID for each subquery
                subquery_id = f"sq_{i:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Create detailed subquery object with metadata
                subquery_obj = {
                    "id": subquery_id,
                    "text": subquery_text.strip(),
                    "order": i,
                    "category": self._categorize_subquery(subquery_text, i),
                    "priority": self._assign_priority(i, len(subqueries_raw)),
                    "estimated_complexity": self._estimate_complexity(subquery_text),
                    "keywords": self._extract_keywords(subquery_text),
                    "metadata": {
                        "created_at": current_timestamp,
                        "agent": "PlannerAgent",
                        "version": "1.0",
                        "parent_query": input_data[:100] + "..." if len(input_data) > 100 else input_data,
                        "character_count": len(subquery_text),
                        "word_count": len(subquery_text.split())
                    }
                }
                structured_subqueries.append(subquery_obj)

            # Log how many subqueries were produced after parsing
            self.log(f"Generated {len(structured_subqueries)} structured subqueries")

            # Combine the query and structured subqueries into a comprehensive result object
            output_data = {
                "original_query": input_data,
                "query_metadata": {
                    "character_count": len(input_data),
                    "word_count": len(input_data.split()),
                    "processed_at": current_timestamp,
                    "agent_version": "PlannerAgent v1.0"
                },
                "subqueries": structured_subqueries,
                "summary": {
                    "total_count": len(structured_subqueries),
                    "categories": self._get_category_summary(structured_subqueries),
                    "priority_distribution": self._get_priority_distribution(structured_subqueries),
                    "avg_complexity": self._calculate_avg_complexity(structured_subqueries)
                },
                "timestamp": current_timestamp
            }

            # Persist both JSON and Markdown representations to disk
            self._save_results(output_data)

            # Return a formatted success payload for consumers
            return self.format_output(output_data)

        except Exception as e:
            # Log any unhandled exceptions and prepare an error payload
            self.log(f"Error in planning: {str(e)}")
            error_data = {
                "error": str(e),
                "original_query": input_data,
                "query_metadata": {
                    "character_count": len(input_data),
                    "word_count": len(input_data.split()),
                    "processed_at": datetime.now().isoformat(),
                    "agent_version": "PlannerAgent v1.0"
                },
                "subqueries": [{
                    "id": f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "text": input_data,
                    "order": 1,
                    "category": "fallback",
                    "priority": "high",
                    "estimated_complexity": "medium",
                    "keywords": input_data.split()[:5],
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "agent": "PlannerAgent",
                        "version": "1.0",
                        "parent_query": input_data,
                        "character_count": len(input_data),
                        "word_count": len(input_data.split()),
                        "is_fallback": True
                    }
                }],
                "summary": {
                    "total_count": 1,
                    "categories": {"fallback": 1},
                    "priority_distribution": {"high": 1},
                    "avg_complexity": "medium"
                },
                "timestamp": datetime.now().isoformat()
            }
            # Attempt to persist the error payload as well
            self._save_results(error_data)
            # Return a standardized error response
            return self.format_output(error_data, status="error")

    def _categorize_subquery(self, subquery_text: str, order: int) -> str:
        """Categorize subquery based on content and order."""
        text_lower = subquery_text.lower()
        
        # Category mapping based on common research patterns
        if order == 1 or any(word in text_lower for word in ['current', 'latest', 'recent', 'state', 'overview']):
            return "current_state"
        elif order == 2 or any(word in text_lower for word in ['companies', 'players', 'organizations', 'leaders']):
            return "key_players"
        elif order == 3 or any(word in text_lower for word in ['market', 'trends', 'adoption', 'economic']):
            return "market_analysis"
        elif order == 4 or any(word in text_lower for word in ['challenges', 'problems', 'barriers', 'limitations']):
            return "challenges"
        elif order == 5 or any(word in text_lower for word in ['future', 'outlook', 'predictions', 'developments']):
            return "future_outlook"
        else:
            return "general"

    def _assign_priority(self, order: int, total_count: int) -> str:
        """Assign priority based on order and total count."""
        if order <= 2:
            return "high"
        elif order <= total_count - 1:
            return "medium"
        else:
            return "low"

    def _estimate_complexity(self, subquery_text: str) -> str:
        """Estimate complexity based on text characteristics."""
        word_count = len(subquery_text.split())
        technical_terms = sum(1 for word in subquery_text.lower().split() 
                            if any(term in word for term in ['technology', 'technical', 'analysis', 'research', 'development']))
        
        if word_count > 30 or technical_terms > 3:
            return "high"
        elif word_count > 15 or technical_terms > 1:
            return "medium"
        else:
            return "low"

    def _extract_keywords(self, subquery_text: str) -> List[str]:
        """Extract key terms from subquery text."""
        # Simple keyword extraction - remove common words and get important terms
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'why', 'when', 'where', 'which', 'that', 'are', 'is', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        
        words = subquery_text.lower().replace('?', '').replace(',', '').replace('.', '').split()
        keywords = [word for word in words if len(word) > 3 and word not in common_words]
        
        # Return top 8 keywords
        return keywords[:8]

    def _get_category_summary(self, subqueries: List[Dict]) -> Dict[str, int]:
        """Get summary of categories."""
        categories = {}
        for sq in subqueries:
            cat = sq.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        return categories

    def _get_priority_distribution(self, subqueries: List[Dict]) -> Dict[str, int]:
        """Get distribution of priorities."""
        priorities = {}
        for sq in subqueries:
            priority = sq.get('priority', 'unknown')
            priorities[priority] = priorities.get(priority, 0) + 1
        return priorities

    def _calculate_avg_complexity(self, subqueries: List[Dict]) -> str:
        """Calculate average complexity."""
        complexity_scores = {'low': 1, 'medium': 2, 'high': 3}
        total_score = sum(complexity_scores.get(sq.get('estimated_complexity', 'medium'), 2) for sq in subqueries)
        avg_score = total_score / len(subqueries) if subqueries else 2
        
        if avg_score <= 1.5:
            return "low"
        elif avg_score <= 2.5:
            return "medium"
        else:
            return "high"

    def _save_results(self, data: Dict[str, Any]) -> None:
        """
        Save results to disk in two formats:
        - JSON: machine-readable structured data
        - Markdown: human-readable formatted report
        """
        try:
            # Build the output directory path (../output/planneroutput relative to this file)
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "output",
                "planneroutput",
            )
            # Create the directory if it doesn't already exist
            os.makedirs(output_dir, exist_ok=True)

            # Use a precise timestamp to avoid filename collisions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Compose and write the JSON file with pretty formatting and UTF-8 encoding
            json_filename = f"planner_result_{timestamp}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Compose and write the Markdown file using a helper formatter
            md_filename = f"planner_result_{timestamp}.md"
            md_path = os.path.join(output_dir, md_filename)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(self._format_markdown(data))

            # Log the final save locations for traceability
            self.log(f"Results saved to: {json_path} and {md_path}")

        except Exception as e:
            # Log any filesystem or serialization errors encountered during saving
            self.log(f"Error saving results: {str(e)}")
    
    def _format_markdown(self, data: Dict[str, Any]) -> str:
        """
        Convert the result payload into a Markdown document for easy reading.
        """
        # Header and overview section
        md_content = f"""# PlannerAgent Results

## Original Query
{data.get('original_query', 'N/A')}

## Query Metadata
- **Character Count**: {data.get('query_metadata', {}).get('character_count', 'N/A')}
- **Word Count**: {data.get('query_metadata', {}).get('word_count', 'N/A')}
- **Processed At**: {data.get('query_metadata', {}).get('processed_at', 'N/A')}
- **Agent Version**: {data.get('query_metadata', {}).get('agent_version', 'N/A')}

## Summary
- **Total Subqueries**: {data.get('summary', {}).get('total_count', 0)}
- **Average Complexity**: {data.get('summary', {}).get('avg_complexity', 'N/A')}

### Category Distribution
"""
        
        # Add category distribution
        categories = data.get('summary', {}).get('categories', {})
        for category, count in categories.items():
            md_content += f"- **{category.replace('_', ' ').title()}**: {count}\n"
        
        md_content += "\n### Priority Distribution\n"
        
        # Add priority distribution
        priorities = data.get('summary', {}).get('priority_distribution', {})
        for priority, count in priorities.items():
            md_content += f"- **{priority.title()}**: {count}\n"

        md_content += "\n## Generated Subqueries\n\n"

        # Append each subquery with detailed information
        subqueries = data.get('subqueries', [])
        
        # Handle both old format (simple strings) and new format (structured objects)
        for i, subquery in enumerate(subqueries, 1):
            if isinstance(subquery, dict):
                # New structured format
                md_content += f"""### {i}. {subquery.get('text', 'N/A')}

**Subquery Details:**
- **ID**: `{subquery.get('id', 'N/A')}`
- **Category**: {subquery.get('category', 'N/A').replace('_', ' ').title()}
- **Priority**: {subquery.get('priority', 'N/A').title()}
- **Estimated Complexity**: {subquery.get('estimated_complexity', 'N/A').title()}
- **Keywords**: {', '.join(subquery.get('keywords', []))}

**Metadata:**
- **Created At**: {subquery.get('metadata', {}).get('created_at', 'N/A')}
- **Character Count**: {subquery.get('metadata', {}).get('character_count', 'N/A')}
- **Word Count**: {subquery.get('metadata', {}).get('word_count', 'N/A')}
- **Agent**: {subquery.get('metadata', {}).get('agent', 'N/A')}
- **Version**: {subquery.get('metadata', {}).get('version', 'N/A')}

---

"""
            else:
                # Legacy format (simple string)
                md_content += f"### {i}. Subquery\n{subquery}\n\n"

        # Add metadata such as timestamp and status
        md_content += f"""## Processing Metadata
- **Timestamp**: {data.get('timestamp', 'N/A')}
- **Status**: {'Error' if 'error' in data else 'Success'}
"""

        # If an error occurred, include the error message for context
        if 'error' in data:
            md_content += f"- **Error**: {data['error']}\n"

        # Return the complete Markdown document
        return md_content


# Interactive test functionality for running this file directly
async def main():
    """Interactive CLI loop to test PlannerAgent end-to-end."""
    # Lazily import environment loader to avoid hard dependency if unused
    from dotenv import load_dotenv
    # Load environment variables from .env if present
    load_dotenv()

    # Print a friendly header
    print("=" * 60)
    print("🤖 PlannerAgent Interactive Test")
    print("=" * 60)
    print()
    print("This tool will help you generate comprehensive subqueries for research.")
    print("The results will be saved in both Markdown and JSON formats.")
    print()

    try:
        # Construct the agent instance
        print("🔄 Initializing PlannerAgent...")
        planner = PlannerAgent()
        print("✅ PlannerAgent initialized successfully!")
        print()

        # Main input loop until the user exits
        while True:
            # Prompt the user for a research query
            print("📝 Please enter your research query:")
            print("   (Type 'quit' or 'exit' to stop)")
            print("-" * 40)

            # Read and normalize input
            user_query = input("Query: ").strip()

            # Exit on common quit keywords
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for using PlannerAgent.")
                break

            # Guard against empty input and continue the loop
            if not user_query:
                print("❌ Please enter a valid query.\n")
                continue

            # Inform the user that processing has started
            print(f"\n🔍 Processing query: '{user_query}'")
            print("⏳ Generating comprehensive subqueries...")
            print()

            try:
                # Run the agent and capture the result payload
                result = await planner.execute(user_query)

                # If successful, print the subqueries and save locations
                if result.get('status') == 'success':
                    print("✅ Subqueries generated successfully!")
                    print(f"📊 Generated {result['data']['count']} subqueries:")
                    print()

                    # Print each subquery in a numbered list
                    for i, subquery in enumerate(result['data']['subqueries'], 1):
                        print(f"   {i}. {subquery}")

                    # Provide a summary of where files were written
                    print()
                    print("💾 Results have been saved to:")
                    print("   📁 backend/output/planneroutput/")
                    print("   📄 Both .json and .md formats")

                else:
                    # Print the error message if the agent indicates failure
                    print("❌ Error occurred during processing:")
                    print(f"   {result.get('data', {}).get('error', 'Unknown error')}")

            except Exception as e:
                # Catch any unexpected exceptions raised during execution
                print(f"❌ Unexpected error: {str(e)}")

            # Visual separator before the next loop iteration
            print("\n" + "=" * 60)
            print()

    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print("\n\n👋 Process interrupted. Goodbye!")
    except Exception as e:
        # Initialization or other top-level errors are reported here
        print(f"\n❌ Failed to initialize PlannerAgent: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Ensure your environment variables are set")
        print("   2. Verify required dependencies are installed: pip install -r requirements.txt")
        print("   3. Check that any local services required by the agent are running")


if __name__ == "__main__":
    # When executed directly, run the interactive main loop
    import asyncio
    asyncio.run(main())