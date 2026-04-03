import time

import requests


class BaseApi:
    session = requests.session()

    def request_send(self, method, url, **kwargs):
        res = self.session.request(method=method, url=url, **kwargs)
        return res

    def file_upload(self, indata, file_path):
        '''
        :param indata:传入参数 如{"knowledge":"1"}
        :param file_path: 上传文件路径
        :return:
        '''
        file_name = file_path.split('\\')[-1]
        files = {'file': (file_name, open(file_path, 'rb'), 'text/plain')}
        res = self.request_send(method='post', url='http://127.0.0.1:3000/api/fileupload', data=indata, files=files)
        return res.json()

        # 通过doc_id删除对应文件接口

    def delete_file(self, indata):
        # indata为doc_id值
        params = {'doc_id': indata}
        res = self.request_send(method='delete', url='http://127.0.0.1:5001/api/files', params=params)
        return res

    # 删除整个知识库标签下文件
    def delete_knowledge(self, payload):
        # payload = {
        #     "knowledgeLabel": "4"
        # }
        resp = self.request_send(method='delete', url='http://127.0.0.1:5001/api/knowledge', params=payload)
        return resp


if __name__ == '__main__':
    indata = {"knowledgeLabel": "1"}
    '''
    # 上传文件
    indata = {"knowledgeLabel": "1"}
    resp = BaseApi().file_upload(indata, file_path='data/positive_texts_3.txt')
    print(resp)
    doc_id = resp['data']['data'][0]['data'][0]['doc_index']

    # 删除文件
    time.sleep(1)
    resp_delete = BaseApi().delete_file(doc_id)
    print(resp_delete.json())
    '''
    # 清空知识库
    resp = BaseApi().delete_knowledge(indata)
    print(resp.status_code)
    print(resp.json())