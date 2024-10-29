import matplotlib.pyplot as plt
import numpy as np

def scatter(x, y, names, c=None, s=None, ax=None, fn=None, fnargs={}, dv=None, **kwargs):
    assert len(x) == len(y) == len(names)
    if c is not None:
        assert len(x) == len(c)
    if ax is None:
        ax = plt.gca()
    if c is None and 'marker' not in kwargs:
        kwargs['marker'] = '.'
    if c is None and 'linestyle' not in kwargs:
        kwargs['linestyle'] = 'none'
    if c is not None and 'alpha' not in kwargs:
            kwargs['alpha'] = .6
    if s is None:
        s = np.full_like(x, plt.rcParams['lines.markersize'] ** 2) #plt.scatter default
    good_inds = np.where(np.isfinite(x + y))[0]
    x = x[good_inds]
    y = y[good_inds]
    s = s[good_inds]
    if c is not None:
        c = c[good_inds]
    names = [names[g] for g in good_inds]
    if type(fn) is list:
        for i in range(len(fnargs)):
            if 'pth' in fnargs[i].keys():
                fnargs[i]['pth'] = [fnargs[i]['pth'][gi] for gi in good_inds]
    if c is None:
        art, = ax.plot(x, y, picker=5, **kwargs)
    else:
        art = ax.scatter(x, y, c=c, s=s, picker=5, cmap='inferno', **kwargs)

    # art=ax.scatter(x,y,picker=5,**kwargs)

    def onpick(event):
        if event.artist == art:
            # ind = good_inds[event.ind[0]]
            ind = event.ind[0]
            print(ind)
            print('onpick scatter: {}: {} ({},{})'.format(ind, names[ind], np.take(x, ind), np.take(y, ind)))
            if dv is not None:
                dv[0] = names[ind]
            if fn is None:
                print('fn is none?')
            elif type(fn) is list:
                for fni, fna in zip(fn, fnargs):
                    fni(names[ind], **fna)
                    # fni(names[ind],**fna,ind=ind)
            else:
                fn(names[ind], **fnargs)

    def on_plot_hover(event):

        for curve in ax.get_lines():
            if curve.contains(event)[0]:
                print('over {0}'.format(curve.get_gid()))

    ax.figure.canvas.mpl_connect('pick_event', onpick)
    return art

def pcolor(x, y, c, mesh=True, ax=None, fn=None, fnargs={}, dv=None, **kwargs):
    #print(x)
    #print(y)
    #assert len(x) == len(y) == len(names)
    #assert len(x) == len(c)
    if ax is None:
        ax = plt.gca()

    if type(fn) is list:
        for i in range(len(fnargs)):
            if 'pth' in fnargs[i].keys():
                fnargs[i]['pth'] = [fnargs[i]['pth'][gi] for gi in good_inds]

    if mesh:
        art = ax.pcolormesh(x, y, c, picker=5, **kwargs)
    else:
        art = ax.pcolor(x, y, c, picker=5, **kwargs)
    x_centers = x[:-1] + np.diff(x[:2]) / 2
    y_centers = y[:-1] + np.diff(y[:2]) / 2
    #print(x_centers)

    nearest = lambda v, p: v[np.argmin(np.abs(v - p))]

    def onpick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

        xc, yc = nearest(x_centers, event.xdata), nearest(y_centers, event.ydata)
        print(f"onpick pcolor close x={xc} y={yc}")

        if fn is None:
            print('fn is none?')
        elif type(fn) is list:
            for fni, fna in zip(fn, fnargs):
                fni(_gx, _gy, **fna)
                # fni(names[ind],**fna,ind=ind)
        else:
            fn(xc, yc, **fnargs)


    ax.figure.canvas.mpl_connect('button_press_event', onpick)
    return art