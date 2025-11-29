"""
FastAPI routes for the evaluation API.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Dict
import uuid

from models.schemas import (
    EvaluationInput,
    EvaluationResult,
    BatchEvaluationRequest,
    DatasetMetrics
)
from evaluation.evaluator import Evaluator
from evaluation.scoring import ScoringAnalyzer
from evaluation.segment_analyzer import SegmentAnalyzer
from metrics.aggregator import MetricsAggregator
from models.database import db
from llm.target_llm import TargetLLM

router = APIRouter()


@router.post("/evaluate", response_model=EvaluationResult)
async def evaluate_output(eval_input: EvaluationInput):
    """
    Evaluate a single target LLM output.
    
    Args:
        eval_input: Evaluation input with prompt and target output
    
    Returns:
        Complete evaluation result
    """
    try:
        evaluator = Evaluator()
        result = await evaluator.evaluate(eval_input)
        
        # Save to database
        await db.save_evaluation(result.model_dump())
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/evaluate/generate-and-test")
async def generate_and_evaluate(
    prompt: str,
    target_provider: str = "openai",
    target_model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 2000
):
    """
    Generate output from target LLM and evaluate it.
    
    Args:
        prompt: User prompt
        target_provider: LLM provider
        target_model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
    
    Returns:
        Evaluation result
    """
    try:
        evaluator = Evaluator()
        result = await evaluator.evaluate_with_target_generation(
            prompt=prompt,
            target_provider=target_provider,
            target_model=target_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Save to database
        await db.save_evaluation(result.model_dump())
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation and evaluation failed: {str(e)}")


@router.post("/evaluate/batch")
async def batch_evaluate(request: BatchEvaluationRequest, background_tasks: BackgroundTasks):
    """
    Batch evaluation of multiple prompts.
    
    Args:
        request: Batch evaluation request
        background_tasks: FastAPI background tasks
    
    Returns:
        Batch job ID and initial status
    """
    batch_id = str(uuid.uuid4())
    
    # Process in background
    background_tasks.add_task(
        _process_batch_evaluation,
        batch_id,
        request.evaluations
    )
    
    return {
        "batch_id": batch_id,
        "status": "processing",
        "total_evaluations": len(request.evaluations)
    }


async def _process_batch_evaluation(batch_id: str, evaluations: List[EvaluationInput]):
    """Background task for batch evaluation."""
    evaluator = Evaluator()
    results = []
    
    for eval_input in evaluations:
        try:
            result = await evaluator.evaluate(eval_input)
            await db.save_evaluation(result.model_dump())
            results.append(result)
        except Exception as e:
            # Log error but continue
            print(f"Error evaluating {eval_input.prompt_id}: {e}")
    
    # Compute and save dataset metrics
    if results:
        metrics = MetricsAggregator.aggregate(results)
        await db.save_dataset_metrics({
            "batch_id": batch_id,
            **metrics.model_dump()
        })


@router.get("/results/{evaluation_id}", response_model=EvaluationResult)
async def get_evaluation_result(evaluation_id: str):
    """
    Retrieve evaluation result by ID.
    
    Args:
        evaluation_id: Evaluation ID
    
    Returns:
        Evaluation result
    """
    result = await db.get_evaluation(evaluation_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return result


@router.get("/results/{evaluation_id}/analysis")
async def get_evaluation_analysis(evaluation_id: str):
    """
    Get detailed analysis of an evaluation.
    
    Args:
        evaluation_id: Evaluation ID
    
    Returns:
        Detailed analysis including risk summary and segment stats
    """
    result = await db.get_evaluation(evaluation_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    # Parse result
    eval_result = EvaluationResult(**result)
    
    # Generate analysis
    risk_summary = ScoringAnalyzer.get_risk_summary(eval_result.judge_output)
    segment_stats = SegmentAnalyzer.get_segment_statistics(
        [seg.model_dump() for seg in eval_result.judge_output.segment_labels]
    )
    
    return {
        "evaluation_id": evaluation_id,
        "risk_summary": risk_summary,
        "segment_statistics": segment_stats,
        "judge_output": eval_result.judge_output
    }


@router.get("/results")
async def list_evaluations(limit: int = 100, skip: int = 0):
    """
    List recent evaluations.
    
    Args:
        limit: Maximum results to return
        skip: Number of results to skip
    
    Returns:
        List of evaluations
    """
    results = await db.get_evaluations(limit=limit, skip=skip)
    return {"evaluations": results, "count": len(results)}


@router.get("/metrics/dataset", response_model=DatasetMetrics)
async def get_dataset_metrics(limit: Optional[int] = None):
    """
    Get aggregated dataset metrics.
    
    Args:
        limit: Optional limit on number of evaluations to include
    
    Returns:
        Dataset-level metrics
    """
    # Get recent evaluations
    evaluations_data = await db.get_evaluations(limit=limit or 1000)
    
    if not evaluations_data:
        raise HTTPException(status_code=404, detail="No evaluations found")
    
    # Parse to EvaluationResult objects
    evaluations = [EvaluationResult(**e) for e in evaluations_data]
    
    # Compute metrics
    metrics = MetricsAggregator.aggregate(evaluations)
    
    return metrics


@router.get("/metrics/summary")
async def get_metrics_summary(limit: Optional[int] = None):
    """
    Get comprehensive metrics summary.
    
    Args:
        limit: Optional limit on evaluations
    
    Returns:
        Metrics and summary statistics
    """
    evaluations_data = await db.get_evaluations(limit=limit or 1000)
    
    if not evaluations_data:
        return {"error": "No evaluations found"}
    
    evaluations = [EvaluationResult(**e) for e in evaluations_data]
    
    metrics = MetricsAggregator.aggregate(evaluations)
    summary_stats = MetricsAggregator.get_summary_stats(evaluations)
    
    return {
        "dataset_metrics": metrics,
        "summary_statistics": summary_stats
    }


@router.post("/target-llm/test")
async def test_target_llm(
    prompt: str,
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 2000
):
    """
    Test target LLM directly without evaluation.
    
    Args:
        prompt: User prompt
        provider: LLM provider
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens
    
    Returns:
        LLM output and metadata
    """
    try:
        llm = TargetLLM(provider=provider, model_name=model)
        result = await llm.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")


@router.post("/evaluate/multi-judge")
async def multi_judge_evaluate(
    eval_input: EvaluationInput,
    judge_configs: List[Dict[str, str]] = None,
    use_super_judge: bool = True,
    super_judge_provider: str = "groq",
    super_judge_model: str = "llama-3.3-70b-versatile"
):
    """
    Evaluate using multiple judges with super judge consensus.
    
    Args:
        eval_input: Evaluation input
        judge_configs: List of judge configs (default: Groq + Perplexity)
        use_super_judge: Whether to use super judge for final consensus
        super_judge_provider: Provider for super judge
        super_judge_model: Model for super judge
    
    Returns:
        Multi-judge evaluation with super judge consensus
    
    Example judge_configs:
        [
            {"provider": "groq", "model": "llama-3.3-70b-versatile"},
            {"provider": "perplexity", "model": "sonar"}
        ]
    """
    from metrics.multi_judge import MultiJudge
    
    try:
        # Default judge configs if not provided
        if judge_configs is None:
            judge_configs = [
                {"provider": "groq", "model": "llama-3.3-70b-versatile"},
                {"provider": "perplexity", "model": "sonar"}
            ]
        
        multi_judge = MultiJudge()
        result = await multi_judge.evaluate_with_multiple_judges(
            eval_input=eval_input,
            judge_configs=judge_configs,
            use_super_judge=use_super_judge,
            super_judge_provider=super_judge_provider,
            super_judge_model=super_judge_model
        )
        
        # Save to database
        await db.save_evaluation({
            "prompt_id": eval_input.prompt_id,
            "multi_judge_result": result,
            "eval_input": eval_input.model_dump()
        })
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-judge evaluation failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LLM Evaluation Framework"
    }
