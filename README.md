# From Zero to Beyond BERT
with pytorch

This repository is for studying natural language processing. There is an implementation for each model here using Pytorch, and we will continue to increase it.

## Installation
```shell
conda env create -f conda_env.yaml
conda activate bert
```

## Data Set

### [대규모 웹데이터 기반 한국어 말뭉치 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=624)

#### Data Format
```json
{
  "SJML": {
    "header": {
      "identifier": "텍스트_웹데이터 말뭉치 구축__7000049128",
      "name": "컨테이너 텍스트 인식을 위한 학습용 데이터셋",
      "category": 0,
      "type": 0,
      "source_file": "BWSC217000049128",
      "source": "0"
    },
    "text": [
      {
        "title": "skt '미디어 추천기술', mwc서 최고 모바일 영상 서비스로 인정받아",
        "subtitle": "",
        "content": "sk텔레콤은 25일(현지시간) 스페인 바르셀로나에서 열린 'gsma 글로벌 모바일 어워드'에서 자사의 'ai 미디어 추천 기술'이 '최고 모바일 영상 서비스(best mobile video content service)' 부문을 수상했다고 26일 밝혔다. . . 글로벌 모바일 어워드는 gsma(세계이동통신사업자협의회)가 주최하는 이동통신 분야 시상식이다. 매년 이동통신 전문가, 애널리스트, 전문 기자로 구성된 심사위원단이 각 분야별 수상자를 선정해 mwc 현장에서 발표·시상한다. . . sk텔레콤과 sk브로드밴드가 지난해 9월부터 미디어 플랫폼에 적용한 'ai 미디어 추천 기술'이 가장 혁신적인 모바일 영상 서비스 및 기술에게 주는 '최고 모바일 영상 서비스' 상을 수상하게 된 것. . . ai 미디어 추천 기술은 영화나 드라마에서 원하는 장면을 골라 볼 수 있도록 찾아주는 '영상분석 기반 장면 검색 기술'과 개인 취향에 따라 콘텐츠를 추천해주는 '콘텐츠 개인화 추천 기술'로 이뤄져 있다. . . (이름) sk텔레콤 미디어랩스장은 'sk텔레콤이 가진 ai 미디어 기술을 미디어 서비스 전반에 확대 적용해 고객의 미디어 시청 경험을 확대하겠다'고 전했다. .",
        "board": "IT|과학|헬스_통신/미디어",
        "writer": "(이름)",
        "write_date": "2019-02-26 14:51:53",
        "url": "https://sample.url",
        "source_site": "뉴스 생산 업체"
      }
    ]
  }
}
```