from selenium import webdriver
import time

def selenium_translator(from_lang, to_lang, sentences):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument('disable-gpu')

    driver = webdriver.Chrome('./chromedriver', chrome_options=options)
    driver.get(f'https://papago.naver.com/?sk={from_lang}&tk={to_lang}')
    
    time.sleep(1)
    
    input_box = driver.find_element_by_css_selector('textarea#txtSource')
    button = driver.find_element_by_css_selector('button#btnTranslate')
    x_button = driver.find_element_by_class_name('btn_text_clse___1Bp8a')
    
    outputs = []
    fail_idx = []
    for idx, sentence in enumerate(sentences):
        try:
            input_box.clear()
            input_box.send_keys(sentence)
            button.click()
            time.sleep(3)
            _q_result = driver.find_element_by_css_selector('div#txtTarget').text
            outputs.append(_q_result)
        except:
            fail_idx.append(idx)
    
    driver.quit()
    return outputs, fail_idx
