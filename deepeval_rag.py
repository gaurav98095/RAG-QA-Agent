"""
RAG Evaluation using DeepEval
Creates custom test cases and evaluates RAG performance
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval import evaluate
from agent import RAGAgent
import json
from typing import List, Dict, Any

class RAGDeepEvaluator:
    def __init__(self, agent: RAGAgent):
        self.agent = agent
        self.metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
        ]
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases for RAG evaluation"""
        return [
            {
                'input': 'How does the Neuralink brain-computer interface work?',
                'expected_output': 'The Neuralink BCI uses ultra-thin threads with electrodes to record neural activity. These threads are surgically implanted by a robot, and the signals are processed by a chip that wirelessly transmits data.',
                'context_keywords': ['interface', 'electrode', 'thread', 'chip', 'wireless', 'neural activity']
            },
            {
                'input': 'What is the current status of Neuralink clinical trials?',
                'expected_output': 'Neuralink has received FDA approval for human trials and has begun recruiting participants for studies focused on helping paralyzed individuals control digital devices.',
                'context_keywords': ['clinical trial', 'FDA approval', 'human trial', 'participant', 'paralyzed', 'digital device']
            },
            {
                'input': 'How is data transmitted from Neuralink devices?',
                'expected_output': 'Data from Neuralink devices is transmitted wirelessly using Bluetooth technology to external devices like smartphones or computers for processing and control.',
                'context_keywords': ['data transmission', 'wireless', 'Bluetooth', 'smartphone', 'computer', 'processing']
            },
            {
                'input': 'What challenges does Neuralink face in development?',
                'expected_output': 'Neuralink faces challenges including regulatory approval, ensuring long-term biocompatibility, scaling manufacturing, addressing ethical concerns, and proving clinical efficacy.',
                'context_keywords': ['challenge', 'regulatory', 'biocompatibility', 'manufacturing', 'ethical', 'efficacy']
            }
        ]
    
    def generate_llm_test_cases(self, test_data: List[Dict[str, Any]]) -> List[LLMTestCase]:
        """Generate LLMTestCase objects from test data"""
        test_cases = []
        
        for data in test_data:
            # Get retrieval context
            retrieved_docs = self.agent.retrieve(data['input'])
            
            # Get agent response
            response = self.agent.answer(data['input'])
            
            # Handle response format
            if isinstance(response, dict):
                actual_output = response.get('answer', str(response))
            else:
                actual_output = str(response)
            
            # Create test case
            test_case = LLMTestCase(
                input=data['input'],
                actual_output=actual_output,
                expected_output=data['expected_output'],
                retrieval_context=retrieved_docs
            )
            
            test_cases.append(test_case)
            print(f"Created test case: {data['input'][:50]}...")
        
        return test_cases
    
    def run_evaluation(self, test_cases: List[LLMTestCase]) -> Dict[str, Any]:
        """Run evaluation using DeepEval metrics"""
        print(f"Running evaluation on {len(test_cases)} test cases...")
        
        # Run evaluation
        results = evaluate(test_cases, self.metrics)
        
        return results
    

    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        print("=== RAG Evaluation with DeepEval ===")
        
        # Create test cases
        print("Creating test cases...")
        test_data = self.create_test_cases()
        
        # Generate LLM test cases
        print("Generating LLM test cases...")
        llm_test_cases = self.generate_llm_test_cases(test_data)
        
        # Run evaluation
        print("Running evaluation...")
        results = self.run_evaluation(llm_test_cases)
        
        print("Evaluation complete!")
        return results


def main():
    """Main evaluation function"""
    try:
        chunking_strategies = [500, 1024, 2048]
        all_results = []
        
        for chunk_size in chunking_strategies:
            print(f"\n=== Evaluating chunk_size={chunk_size} ===")
            
            # Initialize agent with current chunk size
            agent = RAGAgent(document_paths=["neurolink-system.txt"], chunk_size=chunk_size)
            
            # Initialize evaluator
            evaluator = RAGDeepEvaluator(agent)
            
            # Run evaluation
            results = evaluator.run_complete_evaluation()
            
            # Store results
            config_result = {
                'chunk_size': chunk_size,
                'results': str(results)
            }
            all_results.append(config_result)
        
        # Save all results
        with open('chunking_evaluation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n=== All Evaluations Complete ===")
        print("Results saved to chunking_evaluation_results.json")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()