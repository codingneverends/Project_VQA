from flask import Flask, redirect, url_for, request
from flask_cors import CORS, cross_origin
from testmodule import TestModule
app = Flask(__name__)
CORS(app)

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/getresult',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      image = request.files['image']
      print(image)
      qn = request.form['qn']
      val = TestModule(qn)
      return {"status":"OK","content_type":val[0],"question_type":val[1]} , 200

if __name__ == '__main__':
   app.run(host='localhost', port=1212)