from socket import *
import datetime
import concurrent.futures

#KeyencePLCと上位リンク方式で通信するプログラム。
#PORT:8501　
#concurrent.futuresは並行実行モジュール
#実際このClassはroll_ironer_monitor.pyにて使用（20230706現在）

BUFSIZE = 4096

class kvHostLink(object):
    addr = ()
    destfins = []
    srcfins = []

    def __init__(self, host,port=8501):
        self.addr = host,port #タプルにセットする

    # ソケット通信を行う関数
    def communicate(self,command): 
        sock = socket(AF_INET, SOCK_DGRAM)
        sock.settimeout(2)
        try:
            # データ送信
            sock.sendto(command, self.addr)
            # データ受信
            result = sock.recv(BUFSIZE)
            print("Received:", result.decode())
    
        except:
            print("Communicate socket error")
            result = None
            pass

        finally:
            # ソケットをクローズ
            sock.close()
        
        return result
    

    def sendrecive(self,com):
        # ThreadPoolExecutorを作成
        executor = concurrent.futures.ThreadPoolExecutor()
        # 各IPアドレスに対してスレッドを作成し、通信を行う
        for _ in self.addr:
            # 非同期で関数を実行し、結果を受け取る
            future = executor.submit(self.communicate,com)
            # 結果を取得
            result = future.result()
        return result


    #モード切替　0:program　1:run
    def mode(self, mode):
        senddata = 'M' + mode
        rcv = self.sendrecive((senddata + '\r').encode())
        return rcv

    #機種問い合わせ
    def unittype(self):
        rcv = self.sendrecive("?k\r".encode())
        return rcv

    #エラークリア
    def errclr(self):
        senddata = 'ER'
        rcv = self.sendrecive((senddata + '\r').encode())
        return rcv

    #エラー番号確認
    def er(self):
        senddata = '?E'
        rcv = self.sendrecive((senddata + '\r').encode())
        return rcv

    #時刻設定
    def settime(self):
        dt_now = datetime.datetime.now()
        senddata = 'WRT ' + str(dt_now.year)[2:]
        senddata = senddata + ' ' + str(dt_now.month)
        senddata = senddata + ' ' + str(dt_now.day)
        senddata = senddata + ' ' + str(dt_now.hour)
        senddata = senddata + ' ' + str(dt_now.minute)
        senddata = senddata + ' ' + str(dt_now.second)
        senddata = senddata + ' ' + dt_now.strftime('%w')
        rcv = self.sendrecive((senddata + '\r').encode())
        return rcv

    #強制セット１点のみ    
    def set(self, address):
        rcv = self.sendrecive(('ST ' + address + '\r').encode())
        return rcv
    
    #強制リセット１点のみ
    def reset(self, address):
        rcv = self.sendrecive(('RS ' + address + '\r').encode())
        return rcv

    #強制セット連続
    def sts(self, address, num):
        rcv = self.sendrecive(('STS ' + address + ' ' + str(num) + '\r').encode())
        return rcv
    
    #強制リセット連続
    def rss(self, address, num):
        rcv = self.sendrecive(('RSS ' + address + ' ' + str(num) + '\r').encode())
        return rcv

    #データ読出1点のみ
    def read(self, addresssuffix):
        rcv = self.sendrecive(('RD ' + addresssuffix + '\r').encode())
        return rcv

    #データ読出連続
    def reads(self, addresssuffix, num):
        rcv = self.sendrecive(('RDS ' + addresssuffix + ' ' + str(num) + '\r').encode())
        return rcv

    #データ書込１点のみ
    def write(self, addresssuffix, data):
        rcv = self.sendrecive(('WR ' + addresssuffix + ' ' + data + '\r').encode())
        return rcv

    #データ書込連続
    def writs(self, addresssuffix, num, data):
        rcv = self.sendrecive(('WRS ' + addresssuffix + ' ' + str(num) + ' ' + data + '\r').encode())
        return rcv

    #ワードデバイス　モニタ登録
    def mws(self, addresses):
        rcv = self.sendrecive(('MWS ' + addresses + '\r').encode())
        return rcv
    
    #ワードデバイス　モニタ登録したデバイスの読出し
    def mwr(self):
        rcv = self.sendrecive(('MWR\r').encode())
        return rcv

if __name__ == "__main__":
    kv = kvHostLink('192.168.250.1')
    #data = kv.mode('1')
    #print(data)
    #data = kv.er()
    #print(data)
    #data = kv.errclr()
    #print(data)
    #data = kv.unittype()
    #print(data)
    #data = kv.settime()
    #print(data)
    #data = kv.set('MR0')
    #print(data)
    #data = kv.reset('MR1')
    #print(data)
    #data = kv.sts('MR10', 5)
    #print(data)
    #data = kv.rss('MR10', 4)
    #print(data)
    data = kv.read('DM0.U')
    print(data)
    #print(int(data))
    
    #data = kv.reads('DM0.S', 4)
    #print(data)
    #data = kv.write('DM0.U', '2')
    #print(data)
    #data = kv.writs('DM1.S', 4, '1 2 3 4')
    #print(data)
    #data = kv.mws('DM0.H DM1.S DM2.L DM4.U DM5.D')
    #print(data)
    #data = kv.mwr()
    #print(data)