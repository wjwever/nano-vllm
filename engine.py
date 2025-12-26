from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm import SamplingParams


class WsEngine(LLMEngine):
    '''
        this is a wraper of LLMEngine, in order not to modify the original code
    '''

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        return seqs

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, uid: str = ""):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, uid)
        self.scheduler.add(seq)