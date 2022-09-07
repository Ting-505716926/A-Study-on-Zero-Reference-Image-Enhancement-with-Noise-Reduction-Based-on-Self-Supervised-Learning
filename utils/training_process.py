import time

def training_process(step, mean_loss, dict, trainset_size, strat, mode=1):
    # 進度條共20格
    progress_bar_len = 20
    # 進度條每個的大小
    grid_size = trainset_size / progress_bar_len
    
#-------------輸出訓練過程loss-------------- strat
    if mode == 1:
        now = (time.time()) - strat
        
        text = ''
        for index,(key,value) in enumerate(dict.items()):
            mean_loss[index] += value
            text += ' - '
            text +=f"{key}:{value:6f}".format(key = key, value = value)
        
        # 現在第n格
        index = int(step//grid_size)
        print(f'\r{step+1}/{trainset_size} [{"█"*index}{" "*(20-index)}] {"%.2f"% now}s {text}', end='')
        return mean_loss
    
    elif mode == 2:
        now = (time.time()) - strat
        
        text = ''
        for index, key in enumerate(dict.keys()):
            text += ' - '
            text += "{0}:{1:6f}".format(key, mean_loss[index]/trainset_size)
        print(f'\r{step+1}/{trainset_size} [{"█"*progress_bar_len}] {"%.2f"% now}s {text}', end='')
#-------------輸出訓練過程loss----------------------end