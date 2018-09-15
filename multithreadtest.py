import multiprocessing
import threading
import time
import coloredlogs, logging
import sys
import indicators
import pandas as pd
import numpy as np
from queue import *
from tkinter import *
from btfxwss import BtfxWss
import numpy as np
import string as string
import matplotlib.pyplot as plt
import requests
import talib as tb


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc
import datetime as dt


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY

globalz = multiprocessing.Array('i', 32)
varz = []


API_KEY = ""
API_SECRET = ""

class Application(Frame):
    def create_widgets(self):
        self.quit_button = Button(self)
        self.quit_button['text'] = 'STOP'
        self.quit_button['fg'] = 'red'
        self.quit_button['command'] = self.stop #self.quit
        self.quit_button.pack({'side': 'bottom'})

        self.launch_button = Button(self)
        self.launch_button['text'] = 'LAUNCH'
        self.launch_button['fg'] = 'green'
        self.launch_button['command'] = self.launch
        self.launch_button.pack({'side': 'bottom'})

        self.data_button = Button(self)
        self.data_button['text'] = 'CANDLES'
        self.data_button['fg'] = 'BLUE'
        self.data_button['command'] = self.data
        self.data_button.pack({'side': 'bottom'})

        self.sell_button = Button(self)
        self.sell_button['text'] = 'SELL'
        self.sell_button['fg'] = 'YELLOW'
        self.sell_button['command'] = self.sell
        self.sell_button.pack({'side': 'bottom'})

        self.buy_button = Button(self)
        self.buy_button['text'] = 'BUY'
        self.buy_button['fg'] = 'MAGENTA'
        self.buy_button['command'] = self.buy
        self.buy_button.pack({'side': 'bottom'})

        for i in range(32):
            var = StringVar()
            varz.append(var)


        self.labelText = "kekuk"
        self.label0 = Label(root, textvariable=varz[30], width=64, font=("Helvetica", 16))
        self.label0.pack({'side': 'bottom'})
        self.label1 = Label(root, textvariable=varz[0], width=64, font=("Helvetica", 16))
        self.label2 = Label(root, textvariable=varz[1], width=32, font=("Helvetica", 16))
        self.label3 = Label(root, textvariable=varz[2], width=32, font=("Helvetica", 16))
        self.label4 = Label(root, textvariable=varz[3], width=32, font=("Helvetica", 16))
        self.label5 = Label(root, textvariable=varz[4], width=32, font=("Helvetica", 16))
        self.label6 = Label(root, textvariable=varz[5], width=32, font=("Helvetica", 16))
        self.label7 = Label(root, textvariable=varz[6], width=32, font=("Helvetica", 16))
        self.label7.pack({'side': 'bottom'})
        self.label6.pack({'side': 'bottom'})
        self.label5.pack({'side': 'bottom'})
        self.label4.pack({'side': 'bottom'})
        self.label3.pack({'side': 'bottom'})
        self.label2.pack({'side': 'bottom'})
        self.label1.pack({'side': 'bottom'})

        f = Figure(figsize=(5, 4), dpi=100)
        a = f.add_subplot(111)
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2*3.1427*t)
        #a.plot(t, s)
        #self.w = FigureCanvasTkAgg(f, master=root)
        #self.w.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)



        #self.w.pack()
        #self.w.create_line(0, 0, 200, 100)
        #self.w.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))

        #self.w.create_rectangle(50, 25, 150, 75, fill="blue")




    def launch(self):
        launch_flag.value=5

    def stop(self):
        launch_flag.value=-1

    def buy(self):
        launch_flag.value=6

    def sell(self):
        pass

    def data(self):
        #120 candlov max za zapros
        #BTC,ETH,XRP,EOS,ETH,OMG,BCH,IOTA,LTC,NEO,XRP,ELF,EO
        #url = "https://api.bitfinex.com/v2/candles/trade:5m:tBTCUSD/hist?start=1497887100000&end=1497922800000&limit=1000"
        url = "https://api.bitfinex.com/v2/candles/trade:1D:tETCUSD/hist?start=1502496000000&limit=1000"
        #start = datetime.utcnow()

        response = requests.request("GET", url)

        time.sleep(1)

        with open('myRest.ts', 'w') as file:

            #for i in range(len(qq[0][0])):
            #stri = (str(qq[0][0][i]))
            file.write((response.text.replace('],[','\n').replace('[[','').replace(']]','')))
            #file.write("\n")

        time.sleep(1)

        loaded = pd.read_csv('myRest.ts', delimiter=',', names=['Timestamp', 'Open', 'Close', 'High', 'Low', 'Volume'])


        loaded = loaded[::-1]


        #loaded = indicators.randomwalk(loaded)

        #loaded.reindex(indeex=loaded.index[::-1])

        loaded = indicators.HA(loaded)

        #loaded = indicators.chaikin_oscillator(loaded)
        #loaded = indicators.chaikin_oscillator2(loaded)



        #loaded = indicators.moving_average(loaded,10)
        #loaded = indicators.exponential_moving_average(loaded,10)

        #loaded = indicators.acc_dist(loaded)
        #loaded = indicators.chaikin_oscillator(loaded)

        chaikin = tb.ADOSC(high=loaded['HA_High'].values, low=loaded['HA_Low'].values, close=loaded['HA_Close'].values, volume=loaded['Volume'].values, fastperiod=3, slowperiod=10)
        #loaded = indicators.chaikin_oscillator2(loaded)

        loaded = loaded.join(pd.Series(chaikin,name='TALib_Chaikin'))

        #upperband, middleband, lowerband = tb.BBANDS(close=loaded['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        #loaded = loaded.join(pd.Series(upperband,name='BOLLINGER_upper'))
        #loaded = loaded.join(pd.Series(middleband,name='BOLLINGER_middle'))
        #loaded = loaded.join(pd.Series(lowerband,name='BOLLINGER_lower'))

        #loaded = indicators.accumulation_distribution(loaded,0)

        loaded = indicators.nulltest(loaded)

        #loaded = indicators.backtest(loaded)#backtest(loaded)
        #loaded = indicators.backtest_bull(loaded)
        #loaded=np.loadtxt('myRest.ts',delimiter=',')#,ndmin=2)

        #logging.info(type(loaded))

        #logging.info(loaded['Graph'].values)

        #logging.info(loaded.get_value(999, 'Balance'))

        with open('backtest', 'w') as file:
            file.write(loaded.to_string())


        ########################################################### PLOT NUMBER ONE - GRAPH

        #fig = plt.figure()
        ax1 = plt.subplot2grid((1,1), (0,0))

        timestamps = loaded['Timestamp'].values
        timestamps=np.multiply(timestamps, 0.001)

        balancep = loaded['Balance'].values
        openp = loaded['HA_Open'].values
        highp = loaded['HA_High'].values
        lowp = loaded['HA_Low'].values
        closep = loaded['HA_Close'].values
        volumep = loaded['Volume'].values

        x = 0
        ohlc = []
        datep=[mdates.date2num(dt.datetime.fromtimestamp(ts)) for ts in timestamps]
        y = len(datep)

        sizemultiplier = 3*8


        while x < y:
            append_me = datep[x], openp[x], highp[x], lowp[x], closep[x], volumep[x]
            ohlc.append(append_me)

            if loaded.at[x,'Graph']==1:
                plt.text(datep[x], closep[x]+0.4*sizemultiplier, "o", fontsize=12, color='k')
                plt.plot([datep[x], datep[x]], [closep[x], closep[x]+0.35*sizemultiplier], color='k', linestyle='-', linewidth=2)
            else:
                if loaded.at[x,'Graph']==-1:
                    if balancep[x]-balancep[x-1] > 0:
                        coloris = 'g'
                    else:
                        coloris = 'r'
                    plt.text(datep[x], closep[x]+0.4*sizemultiplier, "c"+str(round(balancep[x]-balancep[x-1],3)), fontsize=12, color=coloris)
                    plt.plot([datep[x], datep[x]], [closep[x], closep[x]+0.35*sizemultiplier], color='k', linestyle='-', linewidth=2)
                else:
                    if loaded.at[x,'Graph']==2:
                        plt.text(datep[x], closep[x]+0.4*sizemultiplier, "oS", fontsize=12, color='k')
                        plt.plot([datep[x], datep[x]], [closep[x], closep[x]+0.35*sizemultiplier], color='k', linestyle='-', linewidth=2)
                    else:
                        if loaded.at[x,'Graph']==-2:
                            if balancep[x]-balancep[x-1] < 0:
                                coloris = 'r'
                            else:
                                coloris = 'g'
                            plt.text(datep[x], closep[x]+0.4*sizemultiplier, "cS"+str(round(balancep[x]-balancep[x-1],3)), fontsize=12, color=coloris)
                            plt.plot([datep[x], datep[x]], [closep[x], closep[x]+0.35*sizemultiplier], color='k', linestyle='-', linewidth=2)
            x+=1

        candlestick_ohlc(ax1, ohlc, width=0.02*sizemultiplier, colorup='#77d879', colordown='#db3f3f')

        plt.plot_date(datep, balancep,fmt='-')

        for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax1.grid(True)


        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('ETC/USD, 3h')
        plt.legend()
        plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
        plt.show()


        ########################################################### PLOT NUMBER TWO - REVENUE
        '''
        #fig = plt.figure()
        ax12 = plt.subplot2grid((1,1), (0,0))

        #timestamps = loaded['Timestamp'].values
        #timestamps=np.multiply(timestamps, 0.001)

        #balancep = loaded['Balance'].values

        x = 0
        #y = len(datep)
        ohlc = []
        #datep=[mdates.date2num(dt.datetime.fromtimestamp(ts)) for ts in timestamps]
        #y = len(datep)

        sizemultiplier = 3


        while x < y:
            append_me = datep[x], openp[x], highp[x], lowp[x], closep[x], volumep[x]
            ohlc.append(append_me)

            if loaded.at[x,'Graph']==1:
                plt.text(datep[x], closep[x]+0.4*sizemultiplier, "o", fontsize=12, color='k')
                plt.plot([datep[x], datep[x]], [closep[x], closep[x]+0.35*sizemultiplier], color='k', linestyle='-', linewidth=2)
            else:
                if loaded.at[x,'Graph']==-1:
                    if balancep[x]-balancep[x-1] > 0:
                        coloris = 'g'
                    else:
                        coloris = 'r'
                    plt.text(datep[x], closep[x]+0.4*sizemultiplier, "c"+str(round(balancep[x]-balancep[x-1],3)), fontsize=12, color=coloris)
                    plt.plot([datep[x], datep[x]], [closep[x], closep[x]+0.35*sizemultiplier], color=coloris, linestyle='-', linewidth=2)
            x+=1

        plt.plot_date(datep, balancep)

        #candlestick_ohlc(ax12, ohlc, width=0.02*sizemultiplier, colorup='#77d879', colordown='#db3f3f')

        for label in ax12.xaxis.get_ticklabels():
            label.set_rotation(45)

        ax12.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax12.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax12.grid(True)

        plt.xlabel('Date')
        plt.ylabel('$')
        plt.title('Profit')
        plt.legend()
        plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
        plt.show()
        '''





    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.quit_button = None
        self.pack()
        self.create_widgets()
        self.poll()

    def poll(self):
        """
        This method is required to allow the mainloop to receive keyboard
        interrupts when the frame does not have the focus
        """

        #for i in range(32):
        #    varz[i].set(globalz[i])

        time = globalz[30]+60*60*3
        #varz[30].set("Time: " + str(pd.datetime.utcfromtimestamp(time)) + " ("+ str(time-datetime.utcnow()) + ")")
        time = globalz[0]+60*60*3
        varz[0].set("Timestamp: " + str(pd.datetime.utcfromtimestamp(time)) + " ("+ str(time) + ")")
        varz[1].set("Open: " + str(globalz[1]))
        varz[2].set("Close: " + str(globalz[2]))
        varz[3].set("High: " + str(globalz[3]))
        varz[4].set("Low: " + str(globalz[4]))
        varz[5].set("Volume: " + str(globalz[5]/1000))
        varz[6].set("Wallet: " + str(globalz[6]))

        if varz[31]==1:
            varz[6].set("аагага: " + str(globalz[6]))

        #logging.critical(self.xx.get())
        self.master.after(250, self.poll)
        #, width=100)
        #self.label1.config("sheiit")
        #self.labelText.pack()


        #logging.warning("GLOB.value = %s" % globalz.value)


def worker_function(quit_flag,launch_flag,globalz):
    counter = 0
    established = False
    wss = None
    qq = None
    first = True
    order = False
    OCHLVlast=0
    OCHLVcur=0

    while not quit_flag.value:

        if launch_flag.value == 5:

            if established == False:
                wss = BtfxWss(key= API_KEY, secret= API_SECRET)
                wss.start()

                while not wss.conn.connected.is_set():
                    time.sleep(1)
                    established = True
                    wss.authenticate()
                # Subscribe to some channels
                wss.subscribe_to_ticker('BTCUSD')
                wss.subscribe_to_candles('BTCUSD', '15m')
                wss.subscribe_to_candles('BTCUSD', '1m')
                wss.subscribe_to_order_book('BTCUSD')

            # Do something else
            t = time.time()
            while time.time() - t < 2:
                pass

            # Accessing data stored in BtfxWss:
            ticker_q = wss.candles('BTCUSD', '1m')  # returns a Queue object for the pair.
            wallet=wss.wallets


            while not ticker_q.empty():
                qq=np.asarray(ticker_q.get())
                #globalz[0] = int(np.asarray(qq)[1])
                if type(qq[0][0][0])==list:
                    first = True
                if first:
                    globalz[0] = int(((qq)[0][0][0][0])/1000)
                    #globalz[1] =
                    logging.debug(((qq)[0][0][0][0])/1000)
                    #loggin.debug()
                    #logging.debug(type((qq)[0][0][0][0]))
                    #logging.debug(type((qq)[0][0][0]))
                    #logging.debug(type((qq)[0][0]))
                    #logging.debug(type((qq)[0]))
                    #logging.debug(type((qq)))



                    with open('my.ts', 'w') as file:

                        for i in range(len(qq[0][0])):
                            stri = (str(qq[0][0][i]))
                            file.write((stri))
                            file.write("\n")

                    first = False
                else:
                    globalz[30] = int(((qq)[1]))
                    globalz[0] = int(((qq)[0][0][0])/1000)
                    globalz[5] = int(((qq)[0][0][5])*1000)
                    globalz[6] = int(((qq)[0][0][5])*1000)

                    #globalz[0] =

                    for i in range(1,5):
                        globalz[i] = int((np.asarray(qq)[0][0][i]))

                    #logging.debug(counter)
                    logging.debug((np.asarray(qq)[0][0]))
                    #logging.debug((np.asarray(qq)[0][0][1]))

            if wallet.empty() == False:
                logging.debug(wallet.get())


            #ticker_q = wss.candles('BTCUSD')  # returns a Queue object for the pair.

            counter += 1
            logging.debug("Tick # %s" % counter)

            #var.set()
            #time.sleep(0.01)

        elif launch_flag.value == -1:
            if established == True:
                logging.info("Stopping kekush at # %s" % counter)
                time.sleep(0.01)

                wss.unsubscribe_from_ticker('BTCUSD')
                wss.unsubscribe_from_candles('BTCUSD')
                wss.unsubscribe_from_order_book('BTCUSD')

                # Shutting down the client:
                wss.stop()
                wss = None
                counter = 0
                established = False
                first = True

        elif launch_flag.value == 6:
            if order == False:
                logging.info("Ordering some shit at # %s" % counter)

                order = {
                "cid": 373617,  #change to the API_Key Number
                "type": "LIMIT",
                "symbol": "tEOSUSD",
                "amount": "2",
                "price": "14",
                "hidden": 0,
                "postonly": 1
                }

                wss.new_order(**order)

                launch_flag.value = 7
                order = True

                time.sleep(3)
                logging.info(wss.orders.get())

format = '%(levelname)s: %(filename)s: %(lineno)d: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=format)
coloredlogs.install(level='DEBUG')
root = Tk()
app = Application(master=root)
quit_flag = multiprocessing.Value('i', int(False))
launch_flag = multiprocessing.Value('i', int(False))
launch_flag.value=4
worker_thread = threading.Thread(target=worker_function, args=(quit_flag,launch_flag,globalz,))
worker_thread.start()
logging.info("quit_flag.value = %s" % bool(quit_flag.value))
try:
    app.mainloop()
except KeyboardInterrupt:
    logging.info("Keyboard interrupt")
quit_flag.value = True
logging.info("quit_flag.value = %s" % bool(quit_flag.value))

worker_thread.join()
