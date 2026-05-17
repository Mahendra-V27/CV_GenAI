"""
GenAI Summary Generation Module
Generates natural language summaries using open-source LLMs
Supports: Ollama (local), HuggingFace Transformers, Template fallback
"""


import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import json


# Try to import Ollama
try:
   import ollama
   OLLAMA_AVAILABLE = True
except ImportError:
   OLLAMA_AVAILABLE = False


# Try to import Transformers
try:
   from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
   import torch
   TRANSFORMERS_AVAILABLE = True
except ImportError:
   TRANSFORMERS_AVAILABLE = False




@dataclass
class SummaryResult:
   """Container for generated summary."""
   summary: str
   recommendations: str
   volume_estimate: str
   method: str  # "ollama", "transformers", "template"
   raw_response: Optional[str] = None




class SummaryGenerator:
   """
   Generates natural language inventory summaries using various LLM backends.
  
   Priority:
   1. Ollama (local, open-source)
   2. HuggingFace Transformers (local, open-source)
   3. Template-based (fallback, no LLM)
   """
  
   def __init__(
       self,
       use_ollama: bool = True,
       ollama_model: str = "llama3.2",
       use_transformers: bool = True,
       transformers_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ):
       """
       Initialize summary generator.
      
       Args:
           use_ollama: Try to use Ollama if available
           ollama_model: Ollama model name
           use_transformers: Try to use transformers if Ollama unavailable
           transformers_model: HuggingFace model name
       """
       self.use_ollama = use_ollama and OLLAMA_AVAILABLE
       self.ollama_model = ollama_model
       self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
       self.transformers_model = transformers_model
      
       self._transformers_pipeline = None
       self._ollama_available = None
      
   def _check_ollama_available(self) -> bool:
       """Check if Ollama is running and model is available."""
       if self._ollama_available is not None:
           return self._ollama_available
          
       if not OLLAMA_AVAILABLE:
           self._ollama_available = False
           return False
          
       try:
           # Try to list models
           models = ollama.list()
           model_names = [m.get('name', '').split(':')[0] for m in models.get('models', [])]
           self._ollama_available = self.ollama_model in model_names or \
                                   any(self.ollama_model in n for n in model_names)
           return self._ollama_available
       except Exception as e:
           print(f"Ollama not available: {e}")
           self._ollama_available = False
           return False
  
   def _get_transformers_pipeline(self):
       """Load transformers pipeline (lazy loading)."""
       if self._transformers_pipeline is None and TRANSFORMERS_AVAILABLE:
           try:
               self._transformers_pipeline = pipeline(
                   "text-generation",
                   model=self.transformers_model,
                   torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                   device_map="auto" if torch.cuda.is_available() else None
               )
           except Exception as e:
               print(f"Failed to load transformers model: {e}")
               self._transformers_pipeline = None
       return self._transformers_pipeline
  
   def _build_prompt(
       self,
       inventory: Dict,
       volume_estimate: Dict,
       stats: Dict
   ) -> str:
       """Build the prompt for LLM."""
       # Format inventory items
       items_list = []
       for name, item in inventory.items():
           if isinstance(item, dict):
               items_list.append(f"- {item.get('name', name)}: {item.get('count', 1)} (Category: {item.get('category', 'Other')})")
           else:
               items_list.append(f"- {item.name}: {item.count} (Category: {item.category})")
      
       items_str = "\n".join(items_list) if items_list else "No items detected"
      
       prompt = f"""You are an AI assistant for a moving company. Analyze the following home inventory detected from a video walkthrough and generate a professional summary.


DETECTED INVENTORY:
{items_str}


STATISTICS:
- Total frames processed: {stats.get('total_frames_processed', 0)}
- Unique objects found: {stats.get('unique_objects_found', 0)}
- Unique item types: {stats.get('unique_item_types', 0)}


VOLUME ESTIMATION:
- Total estimated volume: {volume_estimate.get('total_volume_cuft', 0)} cubic feet
- Recommended truck: {volume_estimate.get('recommended_truck', 'Medium')}
- Recommended crew: {volume_estimate.get('recommended_crew', '2-3 persons')}


Please provide:
1. A brief summary of the detected inventory organized by room/category
2. Moving recommendations (truck size, crew, special handling)
3. Packaging suggestions for fragile/large items


Keep the response concise and professional."""


       return prompt
  
   def _generate_with_ollama(self, prompt: str) -> Optional[str]:
       """Generate summary using Ollama."""
       try:
           response = ollama.chat(
               model=self.ollama_model,
               messages=[{"role": "user", "content": prompt}]
           )
           return response['message']['content']
       except Exception as e:
           print(f"Ollama generation error: {e}")
           return None
  
   def _generate_with_transformers(self, prompt: str) -> Optional[str]:
       """Generate summary using HuggingFace Transformers."""
       pipe = self._get_transformers_pipeline()
       if pipe is None:
           return None
          
       try:
           # Format for chat model
           messages = [
               {"role": "system", "content": "You are a helpful assistant for a moving company."},
               {"role": "user", "content": prompt}
           ]
          
           response = pipe(
               messages,
               max_new_tokens=500,
               do_sample=True,
               temperature=0.7,
               top_p=0.9
           )
          
           return response[0]['generated_text'][-1]['content']
       except Exception as e:
           print(f"Transformers generation error: {e}")
           return None
  
   def _generate_template_summary(
       self,
       inventory: Dict,
       volume_estimate: Dict,
       stats: Dict
   ) -> str:
       """Generate summary using templates (fallback, no LLM)."""
       # Organize by category
       by_category = {}
       for name, item in inventory.items():
           if isinstance(item, dict):
               cat = item.get('category', 'Other')
               item_name = item.get('name', name)
               count = item.get('count', 1)
           else:
               cat = item.category
               item_name = item.name
               count = item.count
              
           if cat not in by_category:
               by_category[cat] = []
           by_category[cat].append(f"{count} {item_name.lower()}{'s' if count > 1 else ''}")
      
       # Build summary
       summary_parts = ["## Inventory Summary\n"]
       summary_parts.append("Based on the video walkthrough analysis, your home inventory includes:\n")
      
       for category, items in by_category.items():
           items_str = ", ".join(items)
           summary_parts.append(f"**{category}**: {items_str}")
      
       # Add volume info
       total_vol = volume_estimate.get('total_volume_cuft', 0)
       truck = volume_estimate.get('recommended_truck', 'Medium truck')
       crew = volume_estimate.get('recommended_crew', '2-3 persons')
       boxes = volume_estimate.get('estimated_packing_boxes', 10)
      
       summary_parts.append(f"\n## Moving Recommendations\n")
       summary_parts.append(f"- **Estimated Volume**: {total_vol} cubic feet")
       summary_parts.append(f"- **Recommended Truck**: {truck}")
       summary_parts.append(f"- **Recommended Crew**: {crew}")
       summary_parts.append(f"- **Estimated Packing Boxes**: {boxes} boxes\n")
      
       # Add packaging suggestions
       summary_parts.append("## Packaging Suggestions\n")
      
       fragile_items = ["tv", "laptop", "vase", "clock", "potted plant"]
       large_items = ["sofa", "bed", "refrigerator", "table", "couch"]
      
       has_fragile = any(name.lower() in fragile_items for name in inventory.keys())
       has_large = any(name.lower() in large_items for name in inventory.keys())
      
       if has_fragile:
           summary_parts.append("- **Fragile Items**: Use bubble wrap and specialized boxes for electronics and glassware")
       if has_large:
           summary_parts.append("- **Large Furniture**: Consider furniture blankets and padding for sofas and beds")
       summary_parts.append("- **General**: Label all boxes by room for easier unpacking")
      
       return "\n".join(summary_parts)
  
   def generate_summary(
       self,
       inventory: Dict,
       volume_estimate: Dict,
       stats: Dict
   ) -> SummaryResult:
       """
       Generate natural language summary of the inventory.
      
       Args:
           inventory: Dictionary of InventoryItems
           volume_estimate: Volume estimation data
           stats: Processing statistics
          
       Returns:
           SummaryResult with generated summary
       """
       # Build prompt
       prompt = self._build_prompt(inventory, volume_estimate, stats)
      
       # Try Ollama first
       if self.use_ollama and self._check_ollama_available():
           response = self._generate_with_ollama(prompt)
           if response:
               return SummaryResult(
                   summary=response,
                   recommendations=self._extract_recommendations(response),
                   volume_estimate=f"{volume_estimate.get('total_volume_cuft', 0)} cubic feet",
                   method="ollama",
                   raw_response=response
               )
      
       # Try transformers
       if self.use_transformers:
           response = self._generate_with_transformers(prompt)
           if response:
               return SummaryResult(
                   summary=response,
                   recommendations=self._extract_recommendations(response),
                   volume_estimate=f"{volume_estimate.get('total_volume_cuft', 0)} cubic feet",
                   method="transformers",
                   raw_response=response
               )
      
       # Fallback to template
       template_summary = self._generate_template_summary(inventory, volume_estimate, stats)
       return SummaryResult(
           summary=template_summary,
           recommendations=self._get_template_recommendations(volume_estimate),
           volume_estimate=f"{volume_estimate.get('total_volume_cuft', 0)} cubic feet",
           method="template"
       )
  
   def _extract_recommendations(self, response: str) -> str:
       """Extract recommendations section from LLM response."""
       # Simple extraction - look for recommendation keywords
       lines = response.split('\n')
       rec_lines = []
       in_rec_section = False
      
       for line in lines:
           lower_line = line.lower()
           if 'recommendation' in lower_line or 'suggest' in lower_line:
               in_rec_section = True
           if in_rec_section:
               rec_lines.append(line)
           if in_rec_section and line.strip() == '' and len(rec_lines) > 2:
               break
              
       return '\n'.join(rec_lines) if rec_lines else "See full summary for recommendations."
  
   def _get_template_recommendations(self, volume_estimate: Dict) -> str:
       """Get template-based recommendations."""
       truck = volume_estimate.get('recommended_truck', 'Medium truck')
       crew = volume_estimate.get('recommended_crew', '2-3 persons')
      
       return f"""**Moving Recommendations:**
- Truck: {truck}
- Crew: {crew}
- Schedule: Book 2-3 weeks in advance for best availability
- Duration: Estimate 4-6 hours for packing and loading"""


   def check_availability(self) -> Dict[str, bool]:
       """Check which generation methods are available."""
       return {
           "ollama_installed": OLLAMA_AVAILABLE,
           "ollama_running": self._check_ollama_available() if OLLAMA_AVAILABLE else False,
           "transformers_installed": TRANSFORMERS_AVAILABLE,
           "template_fallback": True  # Always available
       }




# Convenience function for simple usage
def generate_inventory_summary(
   inventory: Dict,
   volume_estimate: Dict,
   stats: Dict
) -> str:
   """
   Simple function to generate inventory summary.
  
   Args:
       inventory: Dictionary of items
       volume_estimate: Volume data
       stats: Processing stats
      
   Returns:
       Generated summary string
   """
   generator = SummaryGenerator()
   result = generator.generate_summary(inventory, volume_estimate, stats)
   return result.summary




if __name__ == "__main__":
   # Test the summary generator
   generator = SummaryGenerator()
  
   # Check availability
   avail = generator.check_availability()
   print("Availability:", avail)
  
   # Test data
   test_inventory = {
       "sofa": {"name": "Sofa", "count": 1, "category": "Furniture", "avg_confidence": 0.9},
       "chair": {"name": "Chair", "count": 4, "category": "Furniture", "avg_confidence": 0.85},
       "tv": {"name": "TV", "count": 1, "category": "Electronics", "avg_confidence": 0.92},
       "bed": {"name": "Bed", "count": 2, "category": "Furniture", "avg_confidence": 0.88},
   }
  
   test_volume = {
       "total_volume_cuft": 180,
       "recommended_truck": "Medium (16-20 ft truck)",
       "recommended_crew": "2-3 persons",
       "estimated_packing_boxes": 60
   }
  
   test_stats = {
       "total_frames_processed": 30,
       "unique_objects_found": 8,
       "unique_item_types": 4
   }
  
   # Generate summary
   result = generator.generate_summary(test_inventory, test_volume, test_stats)
   print(f"\nMethod used: {result.method}")
   print(f"\nSummary:\n{result.summary}")



