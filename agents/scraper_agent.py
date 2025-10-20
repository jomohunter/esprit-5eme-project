from typing import Dict, Any, List
import asyncio
from crawl4ai import AsyncWebCrawler
from .base_agent import BaseAgent

class ScraperAgent(BaseAgent):
    """
    Agent responsible for scraping content from links using Crawl4AI
    """
    
    def __init__(self):
        super().__init__(
            name="ScraperAgent",
            description="Scrapes content from web links using Crawl4AI"
        )
    
    async def execute(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the scraping process
        
        Args:
            input_data: List of query results with links to scrape
            
        Returns:
            Dict containing the scraped content for each link
        """
        try:
            self.log(f"Starting scraping process for {len(input_data)} query results")
            
            all_scraped_results = []
            
            async with AsyncWebCrawler(verbose=False) as crawler:
                for query_result in input_data:
                    original_subquery = query_result["original_subquery"]
                    links = query_result["links"]
                    
                    self.log(f"Scraping {len(links)} links for query: {original_subquery}")
                    
                    scraped_links = []
                    
                    for link in links:
                        scraped_content = await self._scrape_single_link(crawler, link)
                        scraped_links.append(scraped_content)
                    
                    scraped_result = {
                        "original_subquery": original_subquery,
                        "optimized_query": query_result["optimized_query"],
                        "scraped_content": scraped_links,
                        "successful_scrapes": len([s for s in scraped_links if s["status"] == "success"])
                    }
                    
                    all_scraped_results.append(scraped_result)
            
            total_successful = sum(result["successful_scrapes"] for result in all_scraped_results)
            total_attempted = sum(len(result["scraped_content"]) for result in all_scraped_results)
            
            self.log(f"Scraping completed: {total_successful}/{total_attempted} successful")
            
            return self.format_output({
                "results": all_scraped_results,
                "total_queries": len(all_scraped_results),
                "total_links_attempted": total_attempted,
                "total_successful_scrapes": total_successful
            })
            
        except Exception as e:
            self.log(f"Error in scraping process: {str(e)}")
            return self.format_output({
                "error": str(e),
                "results": []
            }, status="error")
    
    async def _scrape_single_link(self, crawler: AsyncWebCrawler, link: Dict[str, str]) -> Dict[str, Any]:
        """
        Scrape content from a single link
        
        Args:
            crawler: The AsyncWebCrawler instance
            link: Dictionary containing link information
            
        Returns:
            Dictionary containing scraped content and metadata
        """
        url = link["url"]
        
        try:
            self.log(f"Scraping: {url}")
            
            # Crawl the URL
            result = await crawler.arun(
                url=url,
                word_count_threshold=10,
                extraction_strategy="NoExtractionStrategy",
                chunking_strategy="RegexChunking",
                bypass_cache=True
            )
            
            if result.success:
                # Extract and clean the content
                content = self._clean_content(result.markdown or result.cleaned_html or "")
                
                scraped_data = {
                    "url": url,
                    "title": link.get("title", ""),
                    "original_snippet": link.get("snippet", ""),
                    "scraped_content": content[:2000],  # Limit content length
                    "content_length": len(content),
                    "status": "success",
                    "timestamp": self._get_timestamp()
                }
                
                self.log(f"Successfully scraped {len(content)} characters from {url}")
                return scraped_data
            
            else:
                self.log(f"Failed to scrape {url}: {result.error_message}")
                return {
                    "url": url,
                    "title": link.get("title", ""),
                    "original_snippet": link.get("snippet", ""),
                    "scraped_content": link.get("snippet", ""),  # Fallback to snippet
                    "content_length": 0,
                    "status": "failed",
                    "error": result.error_message,
                    "timestamp": self._get_timestamp()
                }
        
        except Exception as e:
            self.log(f"Exception while scraping {url}: {str(e)}")
            return {
                "url": url,
                "title": link.get("title", ""),
                "original_snippet": link.get("snippet", ""),
                "scraped_content": link.get("snippet", ""),  # Fallback to snippet
                "content_length": 0,
                "status": "error",
                "error": str(e),
                "timestamp": self._get_timestamp()
            }
    
    def _clean_content(self, content: str) -> str:
        """
        Clean and format scraped content
        
        Args:
            content: Raw scraped content
            
        Returns:
            Cleaned content string
        """
        if not content:
            return ""
        
        # Basic cleaning
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                cleaned_lines.append(line)
        
        # Join lines and limit length
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        import re
        cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content)
        cleaned_content = re.sub(r' +', ' ', cleaned_content)
        
        return cleaned_content.strip()