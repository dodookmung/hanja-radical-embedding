# 한자-부수 임베딩 분석

LLM을 활용해 한자 문자의 부수 기반 구조를 얼마나 잘 포착하는지 측정합니다.

This study examines whether large language models (LLMs) capture the structural meaning of Hanja characters through their radicals. We embed both Hanja and their radicals using a pretrained LLM and measure their similarity. 



## 실험 환경

### Installation

```bash
pip3 install langchain-openai python-dotenv tqdm pandas matplotlib scikit-learn seaborn umap-learn
```

</br></br>

## Acknowledgements

- [rycont/hanja-grade-dataset](https://github.com/rycont/hanja-grade-dataset)  
  프로젝트의 데이터셋을 참고하였습니다. 데이터의 저작권은 한국어문회에 있습니다.
