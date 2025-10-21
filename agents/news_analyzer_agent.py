from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv() 

try:
    from .base_agent import BaseAgent
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base_agent import BaseAgent

class NewsAnalyzerAgent(BaseAgent):
    """
    Agent responsible for analyzing news content and extracting key facts/claims
    """
    
    def __init__(self):
        super().__init__(
            name="NewsAnalyzerAgent",
            description="Extracts key facts and claims from news articles for verification"
        )
        self.fact_extraction_prompt = self._create_fact_extraction_prompt()
        self.fact_chain = LLMChain(llm=self.llm, prompt=self.fact_extraction_prompt)
    
    def _create_fact_extraction_prompt(self) -> PromptTemplate:
        """Create prompt for extracting factual claims from news content"""
        template = """
        Analyze the following news article and extract ALL factual claims, statistics, and key information.
        Focus on quantifiable data, specific claims, numbers, dates, and verifiable statements.

        NEWS CONTENT:
        {content}

        Extract the information in this EXACT JSON format:
        {{
            "key_facts": [
                {{
                    "fact_id": "unique_id_1",
                    "claim": "exact factual claim made",
                    "category": "casualties|economics|politics|health|other",
                    "quantitative_data": {{
                        "numbers": [],
                        "units": "",
                        "timeframe": ""
                    }},
                    "certainty_level": "high|medium|low",
                    "source_attribution": "who is making this claim",
                    "context": "additional context about the claim"
                }}
            ],
            "summary": "brief overall summary of main points",
            "detected_bias_indicators": []
        }}

        Be thorough and extract EVERY factual claim you can find.
        """

        return PromptTemplate(
            input_variables=["content"],
            template=template
        )
    
    async def execute(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze scraped news content and extract factual claims
        
        Args:
            input_data: List of scraped content from ScraperAgent
            
        Returns:
            Dict containing analyzed facts from all articles
        """
        try:
            self.log(f"Analyzing news content from {len(input_data)} sources")
            
            all_analyzed_results = []
            
            for scraped_result in input_data:
                original_subquery = scraped_result["original_subquery"]
                scraped_content = scraped_result["scraped_content"]
                
                self.log(f"Analyzing {len(scraped_content)} articles for: {original_subquery}")
                
                analyzed_articles = []
                
                for article in scraped_content:
                    if article["status"] == "success" and article["scraped_content"]:
                        analysis_result = await self._analyze_single_article(article)
                        analyzed_articles.append(analysis_result)
                
                analyzed_result = {
                    "original_subquery": original_subquery,
                    "analyzed_articles": analyzed_articles,
                    "total_articles_analyzed": len(analyzed_articles)
                }
                
                all_analyzed_results.append(analyzed_result)
            
            # Cross-verify facts across all sources
            cross_analysis = await self._cross_verify_facts(all_analyzed_results)
            
            return self.format_output({
                "analyzed_results": all_analyzed_results,
                "cross_verification": cross_analysis,
                "total_articles_analyzed": sum(len(result["analyzed_articles"]) for result in all_analyzed_results),
                "verification_summary": self._generate_verification_summary(cross_analysis)
            })
            
        except Exception as e:
            self.log(f"Error in news analysis: {str(e)}")
            return self.format_output({
                "error": str(e),
                "analyzed_results": []
            }, status="error")
    
    async def _analyze_single_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single news article and extract facts"""
        try:
            content = article["scraped_content"][:4000]  # Limit content length
            
            # Extract facts using LLM
            analysis_result = await self.fact_chain.arun(content=content)
            
            # Parse the JSON result
            try:
                facts_data = json.loads(analysis_result.strip())
            except json.JSONDecodeError:
                facts_data = {"key_facts": [], "summary": "Failed to parse analysis", "detected_bias_indicators": []}
            
            return {
                "url": article["url"],
                "title": article["title"],
                "source": article.get("source", ""),
                "analysis": facts_data,
                "content_preview": content[:500] + "..." if len(content) > 500 else content,
                "analysis_timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            self.log(f"Error analyzing article {article['url']}: {str(e)}")
            return {
                "url": article["url"],
                "title": article["title"],
                "analysis": {
                    "key_facts": [],
                    "summary": f"Analysis failed: {str(e)}",
                    "detected_bias_indicators": ["analysis_error"]
                },
                "analysis_timestamp": self._get_timestamp()
            }
    
    async def _cross_verify_facts(self, analyzed_results: List[Dict]) -> Dict[str, Any]:
        """Cross-verify facts across different sources to detect discrepancies"""
        try:
            # Collect all facts from all articles
            all_facts = []
            
            for result in analyzed_results:
                for article in result["analyzed_articles"]:
                    for fact in article["analysis"].get("key_facts", []):
                        fact["source_url"] = article["url"]
                        fact["source_title"] = article["title"]
                        all_facts.append(fact)
            
            # Group similar facts by content
            fact_groups = self._group_similar_facts(all_facts)
            
            # Analyze discrepancies
            discrepancy_analysis = self._analyze_discrepancies(fact_groups)
            
            return {
                "total_facts_found": len(all_facts),
                "fact_groups": fact_groups,
                "discrepancies": discrepancy_analysis,
                "consensus_analysis": self._calculate_consensus(fact_groups)
            }
            
        except Exception as e:
            self.log(f"Error in cross-verification: {str(e)}")
            return {
                "total_facts_found": 0,
                "fact_groups": [],
                "discrepancies": [],
                "consensus_analysis": {"error": str(e)}
            }
    
    def _group_similar_facts(self, facts: List[Dict]) -> List[Dict]:
        """Group similar facts together for comparison"""
        # Simple grouping by claim text similarity
        groups = []
        
        for fact in facts:
            matched = False
            claim_lower = fact["claim"].lower()
            
            for group in groups:
                # Check if this fact is similar to existing group
                group_claim_lower = group["representative_claim"].lower()
                
                # Simple similarity check (you could make this more sophisticated)
                common_words = set(claim_lower.split()) & set(group_claim_lower.split())
                if len(common_words) >= 3:  # If they share at least 3 words
                    group["facts"].append(fact)
                    matched = True
                    break
            
            if not matched:
                groups.append({
                    "group_id": f"group_{len(groups)+1}",
                    "representative_claim": fact["claim"],
                    "category": fact["category"],
                    "facts": [fact]
                })
        
        return groups
    
    def _analyze_discrepancies(self, fact_groups: List[Dict]) -> List[Dict]:
        """Analyze discrepancies within fact groups"""
        discrepancies = []
        
        for group in fact_groups:
            if len(group["facts"]) > 1:
                # Extract numerical data for comparison
                numbers_data = []
                
                for fact in group["facts"]:
                    numbers = fact.get("quantitative_data", {}).get("numbers", [])
                    if numbers:
                        numbers_data.append({
                            "source": fact["source_title"],
                            "numbers": numbers,
                            "claim": fact["claim"]
                        })
                
                # Check for numerical discrepancies
                if numbers_data and len(numbers_data) > 1:
                    all_numbers = []
                    for data in numbers_data:
                        all_numbers.extend(data["numbers"])
                    
                    if all_numbers:
                        max_num = max(all_numbers)
                        min_num = min(all_numbers)
                        
                        if max_num > 0 and (max_num - min_num) / max_num > 0.1:  # More than 10% difference
                            discrepancies.append({
                                "group_id": group["group_id"],
                                "claim": group["representative_claim"],
                                "discrepancy_type": "numerical_variance",
                                "variance_percentage": ((max_num - min_num) / max_num) * 100,
                                "range": f"{min_num} to {max_num}",
                                "sources": [{"source": data["source"], "reported_value": data["numbers"]} for data in numbers_data],
                                "severity": "high" if ((max_num - min_num) / max_num) > 0.5 else "medium"
                            })
        
        return discrepancies
    
    def _calculate_consensus(self, fact_groups: List[Dict]) -> Dict[str, Any]:
        """Calculate consensus metrics across all facts"""
        total_facts = sum(len(group["facts"]) for group in fact_groups)
        total_groups = len(fact_groups)
        
        groups_with_consensus = 0
        groups_with_disagreement = 0
        
        for group in fact_groups:
            if len(group["facts"]) > 1:
                # Check if facts in this group are consistent
                numbers_data = []
                for fact in group["facts"]:
                    numbers = fact.get("quantitative_data", {}).get("numbers", [])
                    if numbers:
                        numbers_data.append(numbers[0] if numbers else None)
                
                if numbers_data and len(set(numbers_data)) == 1:
                    groups_with_consensus += 1
                else:
                    groups_with_disagreement += 1
        
        return {
            "total_facts": total_facts,
            "total_groups": total_groups,
            "consensus_rate": (groups_with_consensus / total_groups * 100) if total_groups > 0 else 0,
            "disagreement_rate": (groups_with_disagreement / total_groups * 100) if total_groups > 0 else 0,
            "average_sources_per_fact": total_facts / total_groups if total_groups > 0 else 0
        }
    
    def _generate_verification_summary(self, cross_analysis: Dict) -> str:
        """Generate a human-readable verification summary"""
        discrepancies = cross_analysis.get("discrepancies", [])
        consensus = cross_analysis.get("consensus_analysis", {})
        
        if not discrepancies:
            return "✅ High consensus across all sources. No significant discrepancies detected."
        
        high_severity = len([d for d in discrepancies if d.get("severity") == "high"])
        medium_severity = len([d for d in discrepancies if d.get("severity") == "medium"])
        
        summary = f"🔍 Verification Analysis:\n"
        summary += f"• Found {len(discrepancies)} areas with discrepancies\n"
        summary += f"• {high_severity} high-severity differences\n"
        summary += f"• {medium_severity} medium-severity differences\n"
        summary += f"• Consensus rate: {consensus.get('consensus_rate', 0):.1f}%\n"
        
        if high_severity > 0:
            summary += "⚠️  WARNING: Significant numerical discrepancies detected that require further verification."
        
        return summary