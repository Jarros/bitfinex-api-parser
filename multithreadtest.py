from tkinter import *
import multiprocessing
import threading
import time
import coloredlogs, logging
import sys
import datetime
from pandas import *
import numpy as np
from queue import *
from tkinter import *
from btfxwss import BtfxWss

import mpl_finance as mplf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY

globalz = multiprocessing.Array('i', 32)
varz = []


API_KEY = "your api key here"
API_SECRET = "your secret api key here"

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
        #self.label1.pack()

        self.w = Canvas(self, width=1024, height=512)
        self.w.pack()

        self.w.create_line(0, 0, 200, 100)
        self.w.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))

        self.w.create_rectangle(50, 25, 150, 75, fill="blue")

    def launch(self):
        launch_flag.value=5

    def stop(self):
        launch_flag.value=-1


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
        varz[30].set("Time: " + str(datetime.utcfromtimestamp(time)) + " ("+ str(time) + ")")
        time = globalz[0]+60*60*3
        varz[0].set("Timestamp: " + str(datetime.utcfromtimestamp(time)) + " ("+ str(time) + ")")
        varz[1].set("Open: " + str(globalz[1]))
        varz[2].set("Close: " + str(globalz[2]))
        varz[3].set("High: " + str(globalz[3]))
        varz[4].set("Low: " + str(globalz[4]))
        varz[5].set("Volume: " + str(globalz[5]/1000))
        varz[6].set("Wallet: " + str(globalz[6]))
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
                wss.subscribe_to_candles('BTCUSD')
                wss.subscribe_to_order_book('BTCUSD')

            # Do something else
            t = time.time()
            while time.time() - t < 2:
                pass

            # Accessing data stored in BtfxWss:
            ticker_q = wss.candles('BTCUSD')  # returns a Queue object for the pair.
            wallet=wss.wallets

            while not ticker_q.empty():
                qq=np.asarray(ticker_q.get())
                #globalz[0] = int(np.asarray(qq)[1])
                if first or type(qq[0][0][0])==list:
                    globalz[0] = int(((qq)[0][0][0][0])/1000)
                    #globalz[1] =
                    logging.debug(((qq)[0][0][0][0])/1000)
                    #logging.debug(type((qq)[0][0][0][0]))
                    #logging.debug(type((qq)[0][0][0]))
                    #logging.debug(type((qq)[0][0]))
                    #logging.debug(type((qq)[0]))
                    #logging.debug(type((qq)))
                    for i in range(len(qq[0][0])):
                        logging.debug(qq[0][0][i])

                    with open('my.ts', 'w') as file:
                        file.write('whit')

                    first = False
                else:
                    globalz[30] = int(((qq)[1]))
                    globalz[0] = int(((qq)[0][0][0])/1000)
                    globalz[5] = int(((qq)[0][0][5])*1000)
                    globalz[6] = int(((qq)[0][0][5])*1000)

                    for i in range(1,5):
                        globalz[i] = int((np.asarray(qq)[0][0][i]))

                    #logging.debug(counter)
                    #logging.debug((np.asarray(qq)[0][0][1]))
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

format = '%(levelname)s: %(filename)s: %(lineno)d: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=format)
coloredlogs.install(level='DEBUG')
root = Tk()
app = Application(master=root)
quit_flag = multiprocessing.Value('i', int(False))
launch_flag = multiprocessing.Value('i', int(False))
launch_flag.value=5
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
