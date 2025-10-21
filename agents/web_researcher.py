from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import sys

from dotenv import load_dotenv
load_dotenv()

try:
    from .base_agent import BaseAgent
    from .planner_agent import PlannerAgent
    from .query_generator_agent import QueryGeneratorAgent
    from .link_researcher_agent import LinkResearcherAgent
    from .scraper_agent import ScraperAgent
    from .news_analyzer_agent import NewsAnalyzerAgent
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base_agent import BaseAgent
    from planner_agent import PlannerAgent
    from query_generator_agent import QueryGeneratorAgent
    from link_researcher_agent import LinkResearcherAgent
    from scraper_agent import ScraperAgent
    from news_analyzer_agent import NewsAnalyzerAgent

class WebResearcherAgent(BaseAgent):
    """
    Orchestrates the multi-agent web research workflow
    """
    
    def __init__(self):
        super().__init__(
            name="WebResearcherAgent",
            description="Orchestrates the multi-agent web research workflow"
        )
        
        # Initialize all sub-agents
        self.planner = PlannerAgent()
        self.query_generator = QueryGeneratorAgent()
        self.link_researcher = LinkResearcherAgent()
        self.scraper = ScraperAgent()
        self.news_analyzer = NewsAnalyzerAgent()

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Implement the abstract method from BaseAgent
        """
        if isinstance(input_data, str):
            return await self.research(input_data)
        elif isinstance(input_data, dict) and 'query' in input_data:
            return await self.research(input_data['query'])
        else:
            return await self.research(str(input_data))

    async def research(self, user_query: str) -> Dict[str, Any]:
        """
        Execute the complete research workflow
        
        Args:
            user_query: The research query to investigate
            
        Returns:
            Dict containing comprehensive research results
        """
        try:
            self.log(f"Starting research workflow for: {user_query}")
            
            # Step 1: Plan the research
            self.log("Step 1: Planning research...")
            planning_result = await self.planner.execute(user_query)
            
            if planning_result["status"] != "success":
                return self._format_error_response("Planning failed", planning_result)
            
            subqueries = planning_result["data"]["subqueries"]
            self.log(f"Generated {len(subqueries)} subqueries")
            
            # Step 2: Generate optimized search queries
            self.log("Step 2: Generating search queries...")
            query_gen_result = await self.query_generator.execute(subqueries)
            
            if query_gen_result["status"] != "success":
                return self._format_error_response("Query generation failed", query_gen_result)
            
            search_queries = query_gen_result["data"]["generated"]
            self.log(f"Generated {len(search_queries)} optimized search queries")
            
            # Step 3: Research links for each query
            self.log("Step 3: Researching links...")
            link_research_result = await self.link_researcher.execute(search_queries)
            
            if link_research_result["status"] != "success":
                return self._format_error_response("Link research failed", link_research_result)
            
            link_results = link_research_result["data"]["results"]
            total_links = link_research_result["data"]["total_links"]
            self.log(f"Found {total_links} total links")
            
            # Step 4: Scrape content from links
            self.log("Step 4: Scraping content...")
            scraping_result = await self.scraper.execute(link_results)
            
            if scraping_result["status"] != "success":
                return self._format_error_response("Scraping failed", scraping_result)
            
            scraped_results = scraping_result["data"]["results"]
            successful_scrapes = scraping_result["data"]["total_successful_scrapes"]
            self.log(f"Successfully scraped {successful_scrapes} links")
            
            # Step 5: Analyze news content and verify facts
            self.log("Step 5: Analyzing news content and verifying facts...")
            analysis_result = await self.news_analyzer.execute(scraped_results)
            
            if analysis_result["status"] != "success":
                return self._format_error_response("News analysis failed", analysis_result)
            
            analyzed_data = analysis_result["data"]
            self.log(f"Analyzed {analyzed_data['total_articles_analyzed']} articles")
            
            # Step 6: Compile and summarize results
            self.log("Step 6: Compiling results...")
            final_result = await self._compile_results(
                user_query, 
                subqueries, 
                scraped_results,
                analyzed_data
            )
            
            self.log("Research workflow completed successfully")
            
            return {
                "status": "success",
                "query": user_query,
                "summary": final_result["summary"],
                "detailed_results": final_result["detailed_results"],
                "verification_report": final_result["verification_report"],
                "metadata": {
                    "subqueries_count": len(subqueries),
                    "total_links_found": total_links,
                    "successful_scrapes": successful_scrapes,
                    "articles_analyzed": analyzed_data["total_articles_analyzed"],
                    "discrepancies_found": len(analyzed_data["cross_verification"].get("discrepancies", [])),
                    "processing_steps": [
                        "Planning",
                        "Query Generation", 
                        "Link Research",
                        "Content Scraping",
                        "News Analysis & Verification",
                        "Result Compilation"
                    ]
                }
            }
            
        except Exception as e:
            self.log(f"Error in research workflow: {str(e)}")
            return {
                "status": "error",
                "query": user_query,
                "error": str(e),
                "summary": "Research workflow failed due to an unexpected error.",
                "detailed_results": [],
                "verification_report": {"error": str(e)}
            }

    def _format_error_response(self, message: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response consistently"""
        return {
            "status": "error",
            "error": message,
            "details": result.get("data", {}),
            "summary": f"Research failed: {message}",
            "detailed_results": [],
            "verification_report": {}
        }

    async def _compile_results(self, user_query: str, subqueries: list, scraped_results: list, analyzed_data: Dict = None) -> Dict[str, Any]:
        """
        Compile and summarize the research results with verification
        """
        try:
            # Prepare detailed results
            detailed_results = []
            
            for result in scraped_results:
                subquery_text = result["original_subquery"]
                scraped_content = result["scraped_content"]
                
                successful_content = [
                    content for content in scraped_content 
                    if content["status"] == "success"
                ]
                
                if successful_content:
                    detailed_results.append({
                        "subquery": subquery_text,
                        "sources": [
                            {
                                "title": content["title"],
                                "url": content["url"],
                                "content_preview": content["scraped_content"][:300] + "..." if len(content["scraped_content"]) > 300 else content["scraped_content"],
                                "content_length": content["content_length"]
                            }
                            for content in successful_content
                        ]
                    })
            
            # Include verification report
            verification_report = analyzed_data.get("cross_verification", {}) if analyzed_data else {}
            
            # Generate summary using LLM (enhanced to include verification)
            summary = await self._generate_summary(user_query, detailed_results, verification_report)
            
            return {
                "summary": summary,
                "detailed_results": detailed_results,
                "verification_report": verification_report
            }
            
        except Exception as e:
            self.log(f"Error compiling results: {str(e)}")
            return {
                "summary": f"Research completed for query: {user_query}. Found information across {len(subqueries)} research areas.",
                "detailed_results": [],
                "verification_report": {"error": str(e)}
            }

    async def _generate_summary(self, user_query: str, detailed_results: list, verification_report: Dict = None) -> str:
        """
        Generate a summary of the research results including verification findings
        """
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            # Prepare content for summarization
            content_snippets = []
            for result in detailed_results:
                subquery = result["subquery"]
                for source in result["sources"]:
                    content_snippets.append(f"Topic: {subquery}\nSource: {source['title']}\nContent: {source['content_preview']}")
            
            if not content_snippets:
                return f"Research completed for '{user_query}' but no content was successfully retrieved."
            
            # Enhanced template to include verification
            template = """
            Based on the following research content and verification analysis, provide a comprehensive summary that answers the user's query: "{query}"

            RESEARCH CONTENT:
            {content}

            VERIFICATION FINDINGS:
            {verification}

            Please provide a well-structured summary that:
            1. Directly addresses the user's query
            2. Highlights key findings and insights
            3. Mentions important sources or trends
            4. Notes any discrepancies or consensus issues found during verification
            5. Is concise but informative (2-3 paragraphs)

            Summary:
            """
            
            prompt = PromptTemplate(
                input_variables=["query", "content", "verification"],
                template=template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Limit content length
            combined_content = "\n\n".join(content_snippets[:10])
            if len(combined_content) > 3000:
                combined_content = combined_content[:3000] + "..."
            
            # Prepare verification info
            verification_info = "No verification data available."
            if verification_report:
                discrepancies = verification_report.get("discrepancies", [])
                if discrepancies:
                    verification_info = f"Found {len(discrepancies)} areas with conflicting information. "
                    verification_info += f"Consensus rate: {verification_report.get('consensus_analysis', {}).get('consensus_rate', 0):.1f}%"
                else:
                    verification_info = "High consensus across all sources. No significant discrepancies detected."
            
            summary = await chain.arun(
                query=user_query, 
                content=combined_content,
                verification=verification_info
            )
            
            return summary.strip()
            
        except Exception as e:
            self.log(f"Error generating summary: {str(e)}")
            return f"Research completed for '{user_query}'. Found {len(detailed_results)} relevant research areas."

    async def quick_research(self, user_query: str, max_subqueries: int = 3) -> Dict[str, Any]:
        """
        Perform a quicker research with limited scope
        
        Args:
            user_query: The research query
            max_subqueries: Maximum number of subqueries to process
            
        Returns:
            Dict containing research results
        """
        self.log(f"Starting quick research for: {user_query}")
        
        # Override some agent settings for faster processing
        original_queries_per_subquery = 5  # Store original setting
        
        try:
            # Modify query generator to produce fewer queries for quick research
            self.query_generator.prompt_template = self._create_quick_search_prompt()
            
            planning_result = await self.planner.execute(user_query)
            
            if planning_result["status"] != "success":
                return self._format_error_response("Planning failed", planning_result)
            
            subqueries = planning_result["data"]["subqueries"][:max_subqueries]
            self.log(f"Using {len(subqueries)} subqueries for quick research")
            
            # Continue with normal research workflow but with limited scope
            return await self.research_workflow(user_query, subqueries)
            
        finally:
            # Restore original prompt template
            self.query_generator.prompt_template = self.query_generator._create_prompt_template()

    def _create_quick_search_prompt(self) -> PromptTemplate:
        """Create a prompt for quick research with fewer queries"""
        template = """
        Transform this research subquery into exactly 3 Google search queries.

        Subquery: {subquery}

        REQUIREMENTS:
        - Use professional search operators when helpful
        - Focus on recent information (2024)
        - Each query should be 8-12 words

        RESPOND WITH ONLY A JSON ARRAY - NO OTHER TEXT:
        ["query1", "query2", "query3"]
        """

        from langchain.prompts import PromptTemplate
        return PromptTemplate(
            input_variables=["subquery"],
            template=template
        )


# Interactive test function
async def main():
    """Test the WebResearcherAgent interactively"""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 60)
    print("🔍 Web Researcher Agent Test")
    print("=" * 60)
    
    researcher = WebResearcherAgent()
    
    while True:
        print("\nEnter your research query (or 'quit' to exit):")
        query = input("> ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not query:
            print("Please enter a valid query.")
            continue
            
        print(f"\nResearching: {query}")
        print("This may take a few minutes...")
        
        try:
            result = await researcher.research(query)
            
            if result["status"] == "success":
                print("\n✅ Research completed successfully!")
                print(f"\n📊 Summary: {result['summary'][:500]}...")
                print(f"\n📈 Metadata: {result['metadata']}")
            else:
                print(f"\n❌ Research failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"\n💥 Unexpected error: {str(e)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())