```
from selenium import webdriver
import time

def selenium_translator(from_lang, to_lang, sentences):
```
![image](https://user-images.githubusercontent.com/50571795/135035504-124871d7-3e6e-4747-b44b-98ec80f50f97.png)

1. ```pip install selenium```
2. google에서 chromedriver를 다운받고 설치합니다.
3. chromedriver경로를 파일과 같은 곳에 둡니다.
4. papago에 접속하여, 번역할 언어를 선택합니다. (lang token을 알아야합니다.)
5. ```https://papago.naver.com/?sk=ko&tk=en``` 오른쪽과 같은 주소창의 sk={from_lang}&tk={to_lang}
from_lang과 to_lang을 인자로 줍니다.
6.  원하는 setence를 iterable한 객체로 주시면 됩니다. 