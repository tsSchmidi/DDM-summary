import pandas as pd
import pylab as pl
import math
import random
from tkinter import *
from tkinter import filedialog as fd
import os
from matplotlib import animation
from scipy.optimize import curve_fit


def load_data():
    root = Tk()
    root.withdraw()
    print("Select folder.")
    path = fd.askdirectory()
    root.destroy()

    fitlog = path + "/" + "Fit_log.txt"
    fitlog = pd.read_csv(fitlog, sep="\t", skiprows=range(int(2)), header=0)
    times = fitlog["Time (min)"].values
    times = times - times.min()
    files = fitlog["Fit file name"].values
    files = [path + "/" + file[:-4] + "txt" for file in files]
    data = [pd.read_csv(file, sep="\t", header=None) for file in files]
    columns = ["q", "A", "B", "D", "v", "beta", "Z", "err"]
    data = [df.rename({list(df)[i]:columns[i] for i in range(len(columns))}, axis=1) for df in data]
    data = {times[i]:data[i] for i in range(len(times))}
            
    return data

def Z_to_std(v, Z):
    return v/((Z+1)**0.5)

def summarize(df_dict):
    columns=["time", "nq", "A", "B", "D", "v", "beta", "stdev", "err",
             "dA", "dB", "dD", "dv", "dbeta", "dstdev", "derr"]
    
    summary = []
    for time, df in df_dict.items():
        nq = len(df)
        A = df.A.mean()
        dA = df.A.std()
        B = df.B.mean()
        dB = df.B.std()
        D = df.D.mean()
        dD = df.D.std()
        beta = df.beta.mean()
        dbeta = df.beta.std()
        v_all = df.v * (1 - df.beta)
        v = v_all.mean() / (1 - beta)
        dv = v_all.std() / (1 - beta)
        stdev = df.stdev.mean()
        dstdev = df.stdev.std()
        err = df.err.mean()
        derr = df.err.std()
        summary.append([time, nq, A, B, D, v, beta, stdev, err,
                        dA, dB, dD, dv, dbeta, dstdev, derr])
        
    summary = pd.DataFrame(summary, columns = columns)
    #summary.time = summary.time.sub(summary.time.iloc[0])
    
    return summary

def plot_summary(summary, title=""):
    ignore = ["A", "B", "err", "D", "stdev", "beta"]
    for label in list(summary):
        if "d"+label in list(summary) and label not in ignore:
            pl.figure()
            x = summary.time
            y = summary[label]
            err = summary["d"+label]
            pl.errorbar(x, y, err,
                        fmt="o",
                        ms=3,
                        capsize=3,
                        ecolor="black")
            pl.ylim([0,None])
            pl.xlabel("Time (min)", size=15)
            if label == "v":
                pl.ylabel("Speed (μm/s)", size=15)
            else:
                pl.ylabel(label, size=15)
            pl.title(title, size=15)
    pl.show()

def plot_ddm(df):
    fig, axs = pl.subplots(2, 2, sharex=True)
    axs[0, 0].set_title("Speed")
    axs[0, 0].scatter(df.q, df.v, s=3)
    axs[0, 0].scatter(df.q, df.stdev, s=3)
    axs[0, 0].set_ylim([0, 15])
    
    axs[1, 0].set_title("Non-motile fraction")
    axs[1, 0].scatter(df.q, df.beta, s=3)
    axs[1, 0].set_ylim([0, 1])
    
    axs[0, 1].set_title("Diffusion")
    axs[0, 1].scatter(df.q, df.D, s=3)
    axs[0, 1].set_ylim([0, 0.6])
    
    axs[1, 1].set_title("Signal")
    axs[1, 1].scatter(df.q, df.A, s=3)
    axs[1, 1].scatter(df.q, df.B, s=3)
    axs[1, 1].set_ylim([0, None])
    
    fig.show()

def schulz(xlow, xup, mean, sigma, trim=0):
    xs = pl.linspace(xlow, xup, 10001)[1:]
    z = (mean/sigma)**2 - 1
    ys = [((z+1)**(z+1)*x**z) /
          (mean**(z+1)*math.gamma(z+1)) *
          pl.exp(-(z+1)*x/mean)
          for x in xs]
    
    if trim > 0:
        threshold = trim*sum(ys)
        n_cut = sum([i<threshold for i in pl.cumsum(sorted(ys))])
        y_cut = sorted(ys)[:n_cut]
        ys = [0 if y in y_cut else y for y in ys]
        
    return xs, ys

def size(path):
    coords = path.get_paths()[0].vertices
    length = len(coords)//2
    x = [y for x,y in coords]
    y = [x for x,y in coords]
    pos = (y[1]+y[-2])/2
    x = x[1:length]
    y = y[1:length]
    y = [pos - i for i in y]
    return pl.mean(y)*(max(x)-min(x))

def plot_violin(summary):
    xlow = 2
    xup = 20
    x = summary.time
    y = summary.v
    sigma = summary.stdev
    beta = summary.beta.values
    min_sep = x.diff().min()
    poly = pl.violinplot(
        [random.choices(*schulz(xlow, xup, y[i], sigma[i], trim=0.05), k=10000)
         for i in x.index],
        positions = x,
        showextrema = False)
    pl.clf()
    poly = list(poly.values())[0]
    widths = [size(ob) for ob in poly]
    plotted = [pl.mean(
        [sample > xlow for sample in
         random.choices(*schulz(0, xup, y[i], sigma[i], trim=0.05), k=10000)])
               for i in x.index]
    widths = [(1-beta[i])/widths[i]*plotted[i] for i in range(len(widths))]
    widths = [min_sep*i/max(widths) for i in widths]
    poly = pl.violinplot(
        [random.choices(*schulz(xlow, xup, y[i], sigma[i], trim=0.05), k=10000)
         for i in x.index],
        positions = x,
        widths = widths,
        showextrema = False)
    pl.scatter(x, y, s=3, c="black")
    pl.ylim([0,None])
    pl.show()

def trim(summary, min_sep):
    s = summary.copy()
    while s.time.diff().min() < min_sep:
        s = s.drop(index=s.loc[(s.time.diff() < min_sep)].index[0])
	
    return s.reset_index(drop=True)

def filter_ddm(data,
               qmin = 0.5,
               qmax = 2,
               betamin = 0,
               betamax = 1,
               Dmin = 0,
               Dmax = 1,
               vmin = 0,
               vmax = 30,
               Zmin = -1,
               Amin = 0,
               Bmin = 0,
               errmin = 0):
    data_filtered = data.copy()
    for time in data.keys():
        df = data[time].copy()
        df = df.loc[(df.q.between(qmin, qmax))&
                    (df.beta.between(betamin, betamax))&
                    (df.D.between(Dmin, Dmax))&
                    (df.v.between(vmin, vmax))&
                    (df.Z >= Zmin)&
                    (df.A >= Amin)&
                    (df.B >= Bmin)&
                    (df.err >= errmin)]
        df["stdev"] = df.apply(lambda x: Z_to_std(x.v, x.Z), axis=1)

        if (df.beta[df.q < 1].diff().abs().mean() > 0.08 and
            df.beta[df.q < 1].max() - df.beta[df.q < 1].min() > 0.6 and False):
            df.v = 0.1
            df.dv = 0.1
            df.beta = 0.99
            df.dbeta = 0.01
            print("PNG time " + str(round(time, 1)) + " set to non-motile due to too much variation")
            
        data_filtered[time] = df
    return data_filtered

def plot_effective(summary, title=""):
    df = summary
    x = df.time
    adjust = df.nq / max(df.nq) * (sum(df.beta * df.nq) / sum(df.nq)) / df.beta
    y = df.v * adjust
    err = df.dv * adjust
    fig = pl.figure()
    ax = fig.gca()
    ax.errorbar(x, y, err,
                fmt="o",
                ms=3,
                capsize=3,
                ecolor="black")
    
    #lim = ax.get_xlim()
    #ax.axvspan(4, 69, facecolor = "yellow", alpha = 0.2)
    #ax.axvspan(77, lim[1], facecolor = "yellow", alpha = 0.2)
    #ax.set_xlim([lim[0], lim[1]])
    
    ax.set_ylim([0,None])
    ax.set_xlabel("Time (min)", size=15)
    ax.set_ylabel("Adjusted speed (μm/s)", size=15)
    ax.set_title(title, size=15)
    fig.savefig("fig.png")
    #fig.show()

def plot_motility(summary, title=""):
    global light
    df = summary
    x = df.time
    y = df.v
    err = df.dv# / df.nq**0.5
    y2 = df.beta
    err2 = df.dbeta# / df.nq**0.5
    fig = pl.figure()
    ax = fig.gca()
    ax2 = ax.twinx()
    ax.scatter(x, y, c="navy", s=3)
    ax.fill_between(x,
                    y - err,
                    y + err,
                    color = "navy",
                    alpha = 0.2)
    ax2.scatter(x, 1-y2, c="orangered", s=3)
    ax2.fill_between(x,
                     1 - y2 - err2,
                     1 - y2 + err2,
                     color = "orangered",
                     alpha = 0.2)

    if len(light)%2 == 1:
        light += [summary.time.max()]
    for i_pair in range(len(light)//2):
        ax.axvspan(light[2*i_pair],
                   light[2*i_pair + 1],
                   facecolor = "green", alpha = 0.2)
    for line in vline:
        ax.axvline(line, c="k")
    
    ax.set_ylim([0, None])
    ax2.set_ylim([0, 1])
    ax.set_xlim([0, None])
    ax.tick_params(axis = "both", labelsize = 12)
    ax2.tick_params(axis = "both", labelsize = 12)
    ax.set_xlabel("Time (min)", size=16)
    ax.set_ylabel("Speed (um/s)", size=16, c="navy")
    ax2.set_ylabel("Motile fraction", size=16, c="orangered")
    ax.set_title(title, size=16)
    #pl.subplots_adjust(bottom=0.15, left=0.15, right=0.88)
    fig.savefig("fig.png", dpi=300)
    #fig.show()


def group_speed(summary):
    global light
    df = summary
    inds = [0] + [sum(df.time < i) for i in light] + [len(df)]
    for i in range(len(inds) - 1):
        print("speed")
        print("mean:",df.v[inds[i]:inds[i+1]].mean())
        print("std:",df.v[inds[i]:inds[i+1]].std())
        print("motile")
        print("mean:",1-df.beta[inds[i]:inds[i+1]].mean())
        print("std:",df.beta[inds[i]:inds[i+1]].std())
        print("n:",len(df.v[inds[i]:inds[i+1]]), "\n")

def combine(data, groups = "all"):
    times = list(data.keys())
    qs = list(data.values())[0].q
    data_combined = {}
    if groups == "all":
        groups = [i for i in list(list(data.values())[0]) if i != "q"]
    for group in groups:
        data_combined[group] = pd.concat([df[group] for df in data.values()], axis = 1)
        data_combined[group] = data_combined[group].T
        data_combined[group].columns = qs
        data_combined[group].index = times
        data_combined[group] = data_combined[group].reset_index(names = "time")
    return data_combined

class Animation:
    def __init__(self, data):
        self.data_original = data.copy()
        self.data = data.copy()
        self.data_combined = []

    def filter(self,
               qmin = 0.5,
               qmax = 2,
               betamin = 0,
               betamax = 1,
               Dmin = 0,
               Dmax = 1,
               vmin = 0,
               vmax = 30,
               Zmin = -1,
               Amin = 0,
               Bmin = 0,
               errmin = 0):
        for time in self.data.keys():
            df = self.data[time].copy()
            df = df.loc[(df.q.between(qmin, qmax))&
                        (df.beta.between(betamin, betamax))&
                        (df.D.between(Dmin, Dmax))&
                        (df.v.between(vmin, vmax))&
                        (df.Z >= Zmin)&
                        (df.A >= Amin)&
                        (df.B >= Bmin)&
                        (df.err >= errmin)]
            df["stdev"] = df.apply(lambda x: Z_to_std(x.v, x.Z), axis=1)

            #print(df.beta[df.q < 1].diff().abs().mean(),
                  #df.beta[df.q < 1].max() - df.beta[df.q < 1].min())

            if (df.beta[df.q < 1].diff().abs().mean() > 0.08 and
                df.beta[df.q < 1].max() - df.beta[df.q < 1].min() > 0.6 and False):
                df.v = 0.1
                df.dv = 0.1
                df.beta = 0.99
                df.dbeta = 0.01
                print("GIF time " + str(round(time, 1)) + " set to non-motile due to too much variation")

            self.data[time] = df
        self.qmin = min([df.q.min() for df in self.data.values()])
        self.qmax = max([df.q.max() for df in self.data.values()])

    def interpolate(self, n):
        self.data_combined = combine(self.data_original)
        for group, df in self.data_combined.items():
            df.index = df.index * n
            df = df.reindex(range(df.index.max() + 1))
            df = df.interpolate()
            df = df.melt("time", value_name = group)
            self.data_combined[group] = df
        self.data_combined = pd.concat(
            [df.set_index(["time", "q"]) for df in self.data_combined.values()], axis = 1
            ).reset_index()
        
        if "stdev" not in list(list(self.data.values())[0]):
            temp = self.data.copy()
            self.filter()
            df = summarize(self.data)
            self.data = temp
        else:
            df = summarize(self.data)
        x = df.time
        #adjust = df.nq / max(df.nq) * (sum(df.beta * df.nq) / sum(df.nq)) / df.beta
        y = df.v #* adjust
        err = df.dv #* adjust
        df = pd.DataFrame({"x": x, "y": y, "err": err, "beta": df.beta, "dbeta": df.dbeta, "nq": df.nq})
        df.index = df.index * n
        df = df.reindex(range(df.index.max() + 1))
        df = df.interpolate()
        self.speed = df

    def animate(self, fps):
        if len(self.data_combined) == 0:
            self.interpolate(1)
        fig, axs = pl.subplots(1, 3, figsize = (15, 5), constrained_layout = True)
        ax2 = axs[2].twinx()
        times = self.data_combined.time.unique()
        times = pl.append(times, [max(times)] * fps * 2)
        def update_frame(i):
            global light
            time = times[i]
            time_label = str(math.floor(max([i for i in self.data.keys() if i <= time])))
            df = self.data_combined.loc[self.data_combined.time == time, :]
            axs[0].clear()
            axs[0].set_title("Speed")
            c = ["blue" if self.qmin <= q and q <= self.qmax
                 else "gray" for q in df.q]
            axs[0].scatter(df.q, df.v, s=3, c=c)
            #axs[0].axvspan(self.qmax, lim[1], facecolor = "gray", alpha = 0.2)
            
            axs[0].set_ylim([0, 25])
            axs[0].set_xlabel("q")
            axs[0].set_ylabel("Average swimming speed (um/s)")

            axs[1].clear()
            axs[1].set_title("Motile fraction")
            c = ["orange" if col == "blue" else col for col in c]
            axs[1].scatter(df.q, 1 - df.beta, s=3, c=c)
            if len([t for t in light if t <= time]) % 2 == 1:
                lim = axs[0].get_xlim()
                axs[0].axvspan(lim[0], lim[1],
                               facecolor = "yellow", alpha = 0.2)
                axs[0].set_xlim(lim)
                axs[1].axvspan(lim[0], lim[1],
                               facecolor = "yellow", alpha = 0.2)
                axs[1].set_xlim(lim)
            #axs[1].axvspan(lim[0], self.qmin, facecolor = "gray", alpha = 0.2)
            #axs[1].axvspan(self.qmax, lim[1], facecolor = "gray", alpha = 0.2)
            axs[1].set_ylim([0, 1])
            axs[1].set_xlabel("q")
            axs[1].set_ylabel("Motile fraction")
            
            df = self.speed.loc[self.speed.x <= time, :]
            axs[2].clear()
            ax2.clear()
            axs[2].set_title("Motility")
            axs[2].plot(df.x, df.y, c="blue")
            axs[2].fill_between(df.x,
                                df.y - df.err,# / df.nq**0.5,
                                df.y + df.err,# / df.nq**0.5,
                                color = "blue",
                                alpha = 0.2)
            ax2.plot(df.x, 1 - df.beta, c="orange")
            ax2.fill_between(df.x,
                             1 - df.beta - df.dbeta,# / df.nq**0.5,
                             1 - df.beta + df.dbeta,# / df.nq**0.5,
                             color = "orange",
                             alpha = 0.2)
            
            if len(light)%2 == 1:
                light += [self.speed.x.max()]
            for i_pair in range(len(light)//2):
                axs[2].axvspan(min(light[2*i_pair], time),
                               min(light[2*i_pair + 1], time),
                               facecolor = "yellow", alpha = 0.2)
            for line in vline:
                if time >= line:
                    axs[2].axvline(line, c="k")
                
            axs[2].set_ylim([0, self.speed.y.max()*1.2])
            axs[2].set_xlim([0, self.speed.x.max()*1.05])
            ax2.set_ylim([0, 1])
            axs[2].set_xlabel("time (min)")
            axs[2].set_ylabel("Speed (um/s)", c="blue")
            ax2.set_ylabel("Motile fraction", c="orange")
            
            label = fig.suptitle(time_label + " min",
                            size = 20)
        anim = animation.FuncAnimation(fig, update_frame, frames = len(times))
        anim.save("ani.gif", fps = fps)
        pl.close()


def reverse_sigmoid(x, scale, loc, y_range, y_offset):
    return y_offset + y_range / (1 + pl.exp(scale * x - loc))

    
def depletion_time(summary):
    x = summary.time.copy()
    y = summary.v.copy()
    pl.plot(x, y.diff() / x.diff(), "o")
    pl.show()
    top_index = y.idxmax()
    drop_index = y.diff().idxmin()
    initial_params = [1, x.loc[drop_index], y.max() - y.min(), y.min()]
    pl.plot(x, y, "o")
    #pl.plot(x.loc[top_index:], reverse_sigmoid(x.loc[top_index:], *initial_params))
    pl.show()
    params, cov = curve_fit(reverse_sigmoid,
                            x.loc[top_index:],
                            y.loc[top_index:],
                            initial_params)
    depletion = params[1] / params[0]
    pl.plot(x, y, "o")
    pl.plot(x.loc[top_index:], reverse_sigmoid(x.loc[top_index:], *params))
    print(depletion)
    pl.show()

data = load_data()
light = []
vline = []
time_offset = 3
light = [i + time_offset for i in light]
vline = [i + time_offset for i in vline]
data = {time + time_offset : df for time,df in data.items()}

data_combined = combine(data, ["v", "beta"])

#anim = Animation(data)
#anim.filter(qmin = 0.5, qmax = 2.0)
#anim.interpolate(15)
#anim.animate(15)

data_filtered = filter_ddm(data, qmin = 0.5, qmax = 1.0)
#for df in data_filtered.values():
    #plot_ddm(df)
summary = summarize(data_filtered)
group_speed(summary)
plot_motility(summary, "Title")
