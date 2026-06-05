"""The TestCase: the atomic unit of test data.

Deliberately shape to mirror DeepEval's LLMTestCase' so that the phase-9 migration is a near-1:1
field mapping. The fields cover LLM, RAG, and agent tests so we don't need a different schema
per layer.

"""


from  __future__ import annotations # for forward references in type hints

from pydantic import BaseModel, Field #for data validation and model definition

class TestCase(BaseModel):
    """ A sigle test case. 'input' is the only alwasy-required fields besides id

    Field roles by test layer:
    - LLM: input, actual_output, expected_output
    - RAG: + contex(ground truth) and retrieval_context(what was fetched)
    -Agent: + tools_called / expected_tools (used from phase 5)
    """

    #Not a pytest test class despite the name: stops collection warnings.
    __test__ = False

    # `id` is mandatory for reproducibility and traceability: it seeds per-case RND and is the
    # Primary key in defect reports. A test you can't point at can't be fixed.
    id : str
    #Test IDs are stable, human-meaningful identifiers, not numbers — so they're strings, which keeps them greppable in logs,
    #safe to hash into seeds, and collision-free across merged datasets.

    input : str
    actual_output : str | None = None
    expected_output : str | None = None

    #RAG fields. `context` = thr ground-truth/idea context: 'retrieval_context'
    # = what the retriever actually returned. keeping them separate is what lets
    # us test retrieval quality and generation quality independently.
    context : list[str] | None = None
    retrieval_context : list[str] | None = None

    #Agent fields(Populated in phase 5)
    tools_called: list[str] | None = None
    expected_tools : list[str] | None = None

    #Test-management metadata. Tags drive parameterized/filtered suites.
    # ("run only @safety"), metadata carries provenance(dataset version, etc.).
    tags : list[str] = Field(default_factory=list)
    metadata : dict[str, object] = Field(default_factory=dict)

    model_config = {"frozen": True} #Immutable => Safe to share across async tasks
