from typing import Dict, Any
from .base_agent import BaseAgent
from .planner_agent import PlannerAgent
from .query_generator_agent import QueryGeneratorAgent
from .link_researcher_agent import LinkResearcherAgent
from .scraper_agent import ScraperAgent

class WebResearcherAgent(BaseAgent):
    """
    Main orchestrator agent that coordinates all sub-agents to perform web research
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
    
    async def research(self, user_query: str) -> Dict[str, Any]:
        """
        Execute the complete research workflow
        
        Args:
            user_query: The user's research query
            
        Returns:
            Dict containing the complete research results
        """
        try:
            self.log(f"Starting research workflow for: {user_query}")
            
            # Step 1: Plan the research (break into subqueries)
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
            
            search_queries = query_gen_result["data"]["queries"]
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
            
            # Step 5: Compile and summarize results
            self.log("Step 5: Compiling results...")
            final_result = await self._compile_results(
                user_query, 
                subqueries, 
                scraped_results
            )
            
            self.log("Research workflow completed successfully")
            
            return {
                "status": "success",
                "query": user_query,
                "summary": final_result["summary"],
                "detailed_results": final_result["detailed_results"],
                "metadata": {
                    "subqueries_count": len(subqueries),
                    "total_links_found": total_links,
                    "successful_scrapes": successful_scrapes,
                    "processing_steps": [
                        "Planning",
                        "Query Generation", 
                        "Link Research",
                        "Content Scraping",
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
                "detailed_results": []
            }
    
    async def _compile_results(self, user_query: str, subqueries: list, scraped_results: list) -> Dict[str, Any]:
        """
        Compile and summarize the research results
        
        Args:
            user_query: Original user query
            subqueries: List of subqueries
            scraped_results: Results from scraping
            
        Returns:
            Dict containing compiled results and summary
        """
        try:
            # Prepare detailed results
            detailed_results = []
            
            for result in scraped_results:
                subquery = result["original_subquery"]
                scraped_content = result["scraped_content"]
                
                # Filter successful scrapes
                successful_content = [
                    content for content in scraped_content 
                    if content["status"] == "success"
                ]
                
                if successful_content:
                    detailed_results.append({
                        "subquery": subquery,
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
            
            # Generate summary using LLM
            summary = await self._generate_summary(user_query, detailed_results)
            
            return {
                "summary": summary,
                "detailed_results": detailed_results
            }
            
        except Exception as e:
            self.log(f"Error compiling results: {str(e)}")
            return {
                "summary": f"Research completed for query: {user_query}. Found information across {len(subqueries)} research areas, but encountered issues during result compilation.",
                "detailed_results": []
            }
    
    async def _generate_summary(self, user_query: str, detailed_results: list) -> str:
        """
        Generate a summary of the research results using LLM
        
        Args:
            user_query: Original user query
            detailed_results: Detailed research results
            
        Returns:
            Summary string
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
            
            # Create summarization prompt
            template = """
            Based on the following research content, provide a comprehensive summary that answers the user's query: "{query}"

            Research Content:
            {content}

            Please provide a well-structured summary that:
            1. Directly addresses the user's query
            2. Highlights key findings and insights
            3. Mentions important sources or trends
            4. Is concise but informative (2-3 paragraphs)

            Summary:
            """
            
            prompt = PromptTemplate(
                input_variables=["query", "content"],
                template=template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Limit content length to avoid token limits
            combined_content = "\n\n".join(content_snippets[:10])  # Limit to first 10 snippets
            if len(combined_content) > 3000:
                combined_content = combined_content[:3000] + "..."
            
            summary = await chain.arun(query=user_query, content=combined_content)
            
            return summary.strip()
            
        except Exception as e:
            self.log(f"Error generating summary: {str(e)}")
            return f"Research completed for '{user_query}'. Found {len(detailed_results)} relevant research areas with multiple sources, but encountered issues during summary generation."
    
    def _format_error_response(self, error_message: str, error_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response consistently"""
        return {
            "status": "error",
            "error": error_message,
            "details": error_result,
            "summary": f"Research failed: {error_message}",
            "detailed_results": []
        }