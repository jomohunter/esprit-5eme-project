from typing import Dict, Any, List
import os
import http.client
import json
import asyncio
from .base_agent import BaseAgent

class LinkResearcherAgent(BaseAgent):
    """
    Agent responsible for finding the top 5 relevant links for each search query using Serper.dev
    """
    
    def __init__(self):
        super().__init__(
            name="LinkResearcherAgent",
            description="Finds relevant links using Google search via Serper.dev"
        )
        self.serper_api_key = os.getenv("SERPER_API_KEY")  # ← Remove the hardcoded key
        if not self.serper_api_key:
            self.log("Warning: SERPER_API_KEY not found in environment variables")
    
    async def execute(self, input_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Execute the link research process
        
        Args:
            input_data: List of query dictionaries with 'original_subquery' and 'optimized_query'
            
        Returns:
            Dict containing the found links for each query
        """
        try:
            self.log(f"Researching links for {len(input_data)} queries")
            
            all_results = []
            
            for query_data in input_data:
                optimized_query = query_data["optimized_query"]
                original_subquery = query_data["original_subquery"]
                
                self.log(f"Searching for: {optimized_query}")
                
                # Search using Serper.dev
                links = await self._search_links(optimized_query)
                
                result = {
                    "original_subquery": original_subquery,
                    "optimized_query": optimized_query,
                    "links": links,
                    "link_count": len(links)
                }
                
                all_results.append(result)
                self.log(f"Found {len(links)} links for query: {original_subquery}")
            
            total_links = sum(len(result["links"]) for result in all_results)
            
            return self.format_output({
                "results": all_results,
                "total_queries": len(all_results),
                "total_links": total_links
            })
            
        except Exception as e:
            self.log(f"Error in link research: {str(e)}")
            return self.format_output({
                "error": str(e),
                "results": []
            }, status="error")
    
    async def _search_links(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for links using Serper.dev
        
        Args:
            query: The search query
            num_results: Number of results to return (default: 5)
            
        Returns:
            List of dictionaries containing link information
        """
        try:
            if not self.serper_api_key:
                # Fallback: return mock data for development
                return self._get_mock_links(query, num_results)
            
            # Use Serper.dev API
            links = await self._serper_search(query, num_results)
            return links
            
        except Exception as e:
            self.log(f"Error in Serper.dev search: {str(e)}")
            # Return mock data as fallback
            return self._get_mock_links(query, num_results)
    
    async def _serper_search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Perform search using Serper.dev API
        """
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({
                "q": query,
                "num": num_results
            })
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            # Run the HTTP request in a thread pool to make it async-friendly
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: conn.request("POST", "/search", payload, headers))
            
            response = await loop.run_in_executor(None, conn.getresponse)
            data = await loop.run_in_executor(None, response.read)
            conn.close()
            
            result_data = json.loads(data.decode("utf-8"))
            
            links = []
            if "organic" in result_data:
                for result in result_data["organic"][:num_results]:
                    link_data = {
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "source": result.get("displayLink", "") or result.get("link", "")[:50] + "..."
                    }
                    links.append(link_data)
            
            return links
            
        except Exception as e:
            self.log(f"Serper.dev API error: {str(e)}")
            raise e
    
    def _get_mock_links(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """
        Generate mock links for development/testing purposes
        """
        mock_links = []
        for i in range(min(num_results, 3)):  # Return fewer mock results
            mock_links.append({
                "title": f"Mock Result {i+1} for '{query}'",
                "url": f"https://example{i+1}.com/article-about-{query.replace(' ', '-')}",
                "snippet": f"This is a mock snippet for result {i+1} related to {query}. It contains relevant information about the topic.",
                "source": f"example{i+1}.com"
            })
        
        return mock_links