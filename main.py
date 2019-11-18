import requests
from bs4 import BeautifulSoup

# 第一步：发送第一次请求，获取csrftoken
r1 = requests.get(
    url='https://github.com/login'
)
bs1 = BeautifulSoup(r1.text, 'html.parser')  # 对获取到的文本对象解析获取token值 zlz
obj_token = bs1.find(
    name='input',
    attrs={'name': 'authenticity_token'}
)
# token = obj_token.attrs.get('value')  # 获取token值的两种方式
token = obj_token.get('value')
r1_cookie = r1.cookies.get_dict()
print(r1_cookie)


# 第二步：发送post请求，携带用户名密码并伪造请求头
r2 = requests.post(
    url='https://github.com/session',
    data={
        'commit': 'Sign in',
        'utf8': '✓',
        'authenticity_token': token,
        'login': 'zhouxiangjing',
        'password': '865295386zxj'
    },
    cookies=r1_cookie
)

r2_cookie = r2.cookies.get_dict()
print(r2_cookie)

r1_cookie.update(r2_cookie)
print(r1_cookie)

# 因为是form data提交所以网页是走的重定向，获取状态码&location
# 1、根据状态码；2、根据错误提示

# 第三步：访问个人页面，携带cookie
# r3 = requests.get(
#     url='https://github.com/settings/repositories',
#     cookies=r1_cookie
# )
# print(r3.text)