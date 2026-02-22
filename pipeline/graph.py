from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from .schemas import TextUnderstanding, VisionResponse, SynthesizedOutput, ReflectorResponse, ExitReason
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

text_llm = llm.with_structured_output(TextUnderstanding)
vision_llm = llm.with_structured_output(VisionResponse)
reflector_llm = llm.with_structured_output(ReflectorResponse)
synth_llm = llm.with_structured_output(SynthesizedOutput)

class GraphState(TypedDict):
    page_text: str
    image_base64: str
    context_text: str

    text_understanding: dict
    diagram_description: dict
    
    final_explanation: str
    vision_confidence: float

    synth_confidence: float
    score_history: List[int]
    critique_history: List[str]
    final_quality_score: int
    retry_count: int
    exit_reason: str


def text_node(state: GraphState):
    print("--- Text Node ---")
    prompt = f"""You are a senior researcher. Extract main idea, explain equations conceptually, list key claims.\n\nPAGE TEXT:\n{state['page_text']}"""
    resp = text_llm.invoke([SystemMessage(content=prompt)])
    return {"text_understanding": resp.model_dump()} 

def vision_node(state: GraphState):
    print(f"--- Vision Node (Attempt {state.get('retry_count', 0)}) ---")
    system = "You are a Scientific Vision Analyst. Be precise. Avoid assumptions."

    user_content = [
        {"type": "text", "text": f"Paper context:\n{state['context_text']}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{state['image_base64']}"}}
    ]

    if state["critique_history"]:
        user_content.append({
            "type": "text",
            "text": "FIX THESE ERRORS STRICTLY:\n" + "\n".join(state["critique_history"])
        })

    response = vision_llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_content)
    ])

    return {
        "diagram_description": response.description, # Normalized variable name
        "retry_count": state["retry_count"] + 1,
        "vision_confidence": response.confidence
    }

def reflector_node(state: GraphState):
    print("--- Reflector Node ---")
    prompt = f"""
    PAPER TEXT: {state['context_text']}
    DIAGRAM DESCRIPTION: {state['diagram_description']}
    Evaluate strictly.
    """
    
    response = reflector_llm.invoke([
        SystemMessage(content="You are a ruthless scientific auditor."),
        HumanMessage(content=prompt)
    ])

    # Update histories
    critiques = state["critique_history"]
    if response.critique:
        critiques.append(response.critique)
        
    scores = state["score_history"] + [response.score]

    return {
        "final_quality_score": response.score,
        "score_history": scores,
        "critique_history": critiques
    }

def synth_node(state: GraphState):
    print("--- Synthesis Node ---")
    prompt = f"""Combine these:\nTEXT ANALYSIS: {state['text_understanding']}\nDIAGRAM DESC: {state['diagram_description']}\nProvide a concise research-grade explanation."""
    resp = synth_llm.invoke([SystemMessage(content=prompt)])
    return {"final_explanation": resp.final_explanation}

# Edges
def quality_gate(state: GraphState):
    if state["vision_confidence"] < 0.6:
        return "end_low_conf" # Special routing to END or Synth
        
    if state["final_quality_score"] >= 8:
        return "synth_node" # Proceed to synthesis, not END
        
    if state["retry_count"] >= 3:
        return "synth_node" # Give up and synthesize what we have
        
    if len(state["score_history"]) >= 2:
        if state["score_history"][-1] <= state["score_history"][-2]:
            return "synth_node"

    return "vision_node"

def build_app():
    g = StateGraph(GraphState)
    
    # Add Nodes
    g.add_node("text_node", text_node)
    g.add_node("vision_node", vision_node)
    g.add_node("reflector_node", reflector_node)
    g.add_node("synth_node", synth_node)

    # Flow: Start -> Parallel(Text, Vision)? 
    # For simplicity, let's do Serial: Start -> Text -> Vision -> Loop
    
    g.add_edge(START, "text_node")
    g.add_edge("text_node", "vision_node")
    g.add_edge("vision_node", "reflector_node")

    g.add_conditional_edges(
        "reflector_node",
        quality_gate,
        {
            "vision_node": "vision_node", # Loop
            "synth_node": "synth_node",   # Success/Failover forward
            "end_low_conf": END           # Hard fail
        }
    )
    
    g.add_edge("synth_node", END)

    return g.compile()