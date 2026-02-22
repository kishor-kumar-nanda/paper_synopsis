import os
import operator
from typing import Annotated, List, TypedDict, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

LLM_MODEL = "gpt-4o-mini"

# 1. Define Schemas
class VisionResponse(BaseModel):
    description: str = Field(..., description="Detailed technical description of the diagram.")
    key_elements: List[str] = Field(..., description="List of labels, axes, or key figures identified.")

class ReflectorResponse(BaseModel):
    is_accurate: bool = Field(..., description="True if the description matches the text context.")
    score: int = Field(..., description="Quality score from 1-10.")
    critique: str = Field(..., description="Specific instructions on what to fix. Empty if accurate.")
    missing_info: List[str] = Field(default_factory=list, description="List of concepts mentioned in text but missing in diagram description.")

# 2. Define State
class GraphState(TypedDict):
    image_base64: str
    context_text: str
    current_description: str
    critique_history: List[str]
    retry_count: int
    final_quality_score: int

# 3. Define Models
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
vision_llm = llm.with_structured_output(VisionResponse)
reflector_llm = llm.with_structured_output(ReflectorResponse)

# 4. Define Nodes
def vision_node(state: GraphState):
    """
    Role: Analyze the image.
    Behavior: Looks at the image + previous critiques to generate a description.
    """
    attempt = state["retry_count"] + 1
    print(f"\n [Vision Node] Generating Description (Attempt {attempt})...")

    system_msg = "You are a Scientific Vision Analyst. accurate Extract data, structural relationships, and labels."
    
    user_content = [
        {"type": "text", "text": f"Context from paper: {state['context_text']}"}
    ]
    
    # Add image (In real use, you'd pass the actual base64 string here)
    # For this code to run without an actual image file, I am commenting the image block out.
    # user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_base64']}"}})
    user_content.append({"type": "text", "text": "[IMAGE DATA WOULD BE HERE]"})

    # If we have previous critique, inject it forcefully
    if state["critique_history"]:
        history_text = "\n".join(state["critique_history"])
        user_content.append({
            "type": "text", 
            "text": f"CRITICAL INSTRUCTION: Your previous attempts failed. Fix these specific errors:\n{history_text}"
        })

    # Invoke
    response: VisionResponse = vision_llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=user_content)
    ])
    
    return {
        "current_description": response.description,
        "retry_count": state["retry_count"] + 1
    }

def reflector_node(state: GraphState):
    """
    Role: Quality Control.
    Behavior: Reads the generated description AND the paper text. Checks for hallucinations.
    """
    print(f"🔸 [Reflector Node] Validating against text context...")
    
    prompt = f"""
    Verify this diagram description against the paper's text.
    
    PAPER TEXT:
    {state['context_text']}
    
    GENERATED DESCRIPTION:
    {state['current_description']}
    
    Task:
    1. Does the description hallucinate data not present or implied?
    2. Did it miss key labels mentioned in the text?
    """
    
    response: ReflectorResponse = reflector_llm.invoke([
        SystemMessage(content="You are a strict QA Auditor for Scientific Data."),
        HumanMessage(content=prompt)
    ])
    
    print(f"   -> Score: {response.score}/10 | Accurate: {response.is_accurate}")
    
    new_critiques = state["critique_history"]
    if response.critique:
        new_critiques.append(response.critique)
        
    return {
        "final_quality_score": response.score,
        "critique_history": new_critiques
    }

# 4. Define routering 
def quality_gate(state: GraphState):
    """
    Decides: Loop back to Vision OR Exit?
    """
    score = state["final_quality_score"]
    retries = state["retry_count"]

    if score >= 8:
        print("Quality Standard Met. Finalizing.")
        return END

    if retries >= 3:
        print("Max Retries Reached. Proceeding with best effort.")
        return END

    print("Quality low. Retrying with feedback...")
    return "vision_node"

# --- 5. Build Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("vision_node", vision_node)
workflow.add_node("reflector_node", reflector_node)

workflow.add_edge(START, "vision_node")
workflow.add_edge("vision_node", "reflector_node")
workflow.add_conditional_edges(
    "reflector_node",
    quality_gate,
    {
        "vision_node": "vision_node",
        END: END
    }
)

# Compile
app = workflow.compile()

# --- 6. Execution Example ---

if __name__ == "__main__":
    # Simulated Input (As if coming from your PDF Pipeline)
    initial_state = {
        "image_base64": "dummy_base64_string_xyz", 
        "context_text": "Figure 3 illustrates the linear relationship between voltage and current. The slope represents a resistance of 50 Ohms.",
        "critique_history": [],
        "retry_count": 0,
        "current_description": "",
        "final_quality_score": 0
    }

    print("🚀 Starting Pipeline...")
    final_state = app.invoke(initial_state)
    
    print("\n--- FINAL RESULT ---")
    print(f"Description: {final_state['current_description']}")
    print(f"Final Score: {final_state['final_quality_score']}")