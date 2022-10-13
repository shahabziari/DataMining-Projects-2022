def interpolation(data, config):
    if config['time'] == 'daily':
        data = data.set_index('time')
        data = data.resample('D')
        if (config['interpolation']=="spline" or "polynomial"):
            data = data.interpolate(method=config['interpolation'] , order = config['order'])
        else : data = data.interpolate(method=config['interpolation'])
        data.reset_index(inplace=True)

    elif config['time'] == 'monthly':
        data = data.set_index('time')
        data = data.resample('M')
        if (config['interpolation']=="spline" or "polynomial"):
            data = data.interpolate(method=config['interpolation'] , order = config['order'])
        else : data = data.interpolate(method=config['interpolation'])
        data.reset_index(inplace=True)

    else:
        data = None

    return data
