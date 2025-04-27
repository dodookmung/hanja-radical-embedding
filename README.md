# 한자-부수 임베딩 분석

LLM을 활용해 한자 문자의 부수 기반 구조를 얼마나 잘 포착하는지 측정합니다.

This study examines whether large language models (LLMs) capture the structural meaning of Hanja characters through their radicals. We embed both Hanja and their radicals using a pretrained LLM and measure their similarity. 


## 실험 환경

### Installation

```bash
pip3 install langchain-openai python-dotenv tqdm pandas matplotlib scikit-learn seaborn umap-learn
```

## 실험 결과

LLM을 이용하여 각 한자와 부수를 임베딩한 후, 고차원 임베딩 벡터를 2차원 배열로 변환하고, 이들 간의 코사인 유사도를 계산하였습니다.

### Model Benchmarks

<table>
  <tr>
    <th>Builder</th>
    <th>Model</th>
    <th>ROC-AUC</th>
    <th>TPR</th>
    <th>FPR</th>
  </tr>
  <tr>
    <td rowspan="4">OpenAI<td>
  </tr>
  <tr>
    <td>text-embedding-3-large</td>
    <td>0.7228</td>
    <td>0.6089</td>
    <td>0.2830</td>
  </tr>
  <tr>
    <td>text-embedding-3-small</td>
    <td>0.7245</td>
    <td>0.5515</td>
    <td>0.2076</td>
  </tr>
  <tr>
    <td>text-embedding-ada-002</td>
    <td>0.7167</td>
    <td>0.5169</td>
    <td>0.1937</td>
  </tr>
</table>


</br></br>


### Hugging Face Models

<table>
  <tr>
    <th>Builder</th>
    <th>Model</th>
    <th>ROC-AUC</th>
    <th>TPR</th>
    <th>FPR</th>
  </tr>
  <tr>
    <td>jhgan</td>
    <td>ko-sroberta-multitask</td>
    <td>0.4018</td>
    <td>0.0028</td>
    <td>0.0000</td>
  </tr>
  <tr>
    <td rowspan="5">sentence-transformers<td>
  </tr>
  <tr>
    <td>all-MiniLM-L6-v2</td>
    <td>0.3527</td>
    <td>0.0018</td>
    <td>0.0000</td>
  </tr>
  <tr>
    <td>paraphrase-multilingual-MiniLM-L12-v2</td>
    <td>0.4627</td>
    <td>0.9587</td>
    <td>0.9321</td>
  </tr>
  <tr>
    <td>paraphrase-multilingual-mpnet-base-v2</td>
    <td>0.4853</td>
    <td>0.8914</td>
    <td>0.8533</td>
  </tr>
  <tr>
    <td>distiluse-base-multilingual-cased-v1</td>
    <td>0.5202</td>
    <td>0.6619</td>
    <td>0.5848</td>
  </tr>
  <tr>
    <td>embaas</td>
    <td>sentence-transformers-multilingual-e5-large</td>
    <td>0.5983</td>
    <td>0.3670</td>
    <td>0.2170</td>
  </tr>
</table>

</br></br>

## Acknowledgements

- [rycont/hanja-grade-dataset](https://github.com/rycont/hanja-grade-dataset)  
  프로젝트의 데이터셋을 참고하였습니다. 데이터의 저작권은 한국어문회에 있습니다.
