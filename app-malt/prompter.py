from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from prompt_agent import BasePromptAgent, ZeroShot_CoT_PromptAgent, \
    FewShot_Basic_PromptAgent, FewShot_Semantic_PromptAgent, ReAct_PromptAgent

prompt_suffix = """Begin! Remember to ensure that you generate valid Python code in the following format:

        Answer:
        ```python
        ${{Code that will answer the user question or request}}
        ```
        Question: {input}
        """
        
class PromptAgent: 
    def __init__(self, prompt_type="base", local=False): 
        self.prompt_type = prompt_type 
        self.local = local 
        
        # Generate appropriate agent for prompt_type
        if self.prompt_type == "cot":
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = BasePromptAgent()
    
    def get_prompt(self, query): 
        print("\nPrompt Template:")
        print("-" * 80)
        
        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(query)
            self.print_fs_template(prompt)
            
            if self.local: 
                # For few-shot semantic, we need to format with the specific query
                prompt = prompt.format(input=query)
        elif self.prompt_type == "few_shot_basic":
            prompt = self.prompt_agent.get_few_shot_prompt()
            self.print_fs_template(prompt)
            
            if self.local: 
                # For few-shot basic, format with the query
                prompt = prompt.format(input=query)
        else:
            # For base/cot prompts
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
            )
            
            print(prompt.template.strip())
            
            if self.local: 
                prompt = self.prompt_agent.prompt_prefix + prompt_suffix
                prompt = prompt.format(input=query)
        print("-" * 80 + "\n")
        
        return prompt
    
    def print_fs_template(self, prompt): 
        print("Few Shot Prompt Template Configuration:")
        print("\nInput Variables:", prompt.input_variables)
        print("\nExamples:")
        for i, example in enumerate(prompt.examples, 1):
            print(f"\nExample {i}:")
            print(f"Question: {example['question']}")
            print(f"Answer: {example['answer']}")
        print("\nExample Prompt Template:", prompt.example_prompt)
        print("\nPrefix:", prompt.prefix)
        print("\nSuffix:", prompt.suffix)