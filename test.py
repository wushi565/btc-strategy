from zeep import Client

# WSDL 地址
wsdl = 'http://www.hzsbgs.com:9000/d/csl/Xb.wsdl'

# 初始化客户端
client = Client(wsdl=wsdl)

# 构造参数
params = {
    'cusNo': '',                # 留空表示查询所有（或填具体编号如 "123456"）
    'user': 'sxpgx',
    'pwd': 'sxpgx123456'
}

# 调用接口
response = client.service.getBaInfo(**params)

# 打印原始响应
print(response)
