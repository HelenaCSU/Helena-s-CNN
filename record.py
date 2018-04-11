import time as t

class OfTime():
    def __init__(self):
        self.unit=['年', '月', '日', '时', '分', '秒']
        self.prompt="not start"
        self.start=0
        self.stop=0
        self.lasted=[]

    def __str__(self):
        return self.prompt # 通过重载__str__方法，来确保str(OfTime)输出当前运行的时间

    __repr__=__str__

    # when we select a point in the video, start the timer
    def start(self):
        self.start=t.localtime()
        self.prompt="stop at first"
        print("start recording")
    # when we detect the direction of point reverse, stop the timer
    def stop(self):
        if not self.start():
            print("start at first")
        else:
            self.stop=t.localtime()
            self._calc()
        print("stop recording")

    def _calc(self):
        self.lasted=[]
        self.prompt="total time"
        for index in range(6):
            self.lasted.append(self.stop[index]-self.start[index])
            self.prompt +=str(self.lasted[index])