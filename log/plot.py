import matplotlib.pyplot as plt

def plot_log(log_name):

    index = []
    loss = []
    acc = []
    dice = []
    valid_idx = []
    valid_acc = []
    valid_dice = []
    with open(log_name, 'r') as f:
        lines = f.readlines()
        
        for idx, line in enumerate(lines):
            if line[:5] == "Batch":
                data = line[:-1].split(',')
                data[3] = data[3].strip(' ')

                index.append(int(data[0].split(':')[1]))
                loss.append(float(data[1][-8:]))
                acc.append(float(data[2][-7:]))
                dice.append(float(data[3][-7:]))

            if idx+1<len(lines) and lines[idx+1][:5] == "valid" and line[:5] == "Batch":
                valid_idx.append(int(line.split(' ')[1][:-1]))

            elif line[:5] == "valid":

                data = line.strip('\n').split(' ')
                if(data[1]=='acc:'):
                    valid_acc.append(float(data[2]))
                elif(data[1] == 'dice:'):
                    valid_dice.append(float(data[2]))


    plt.figure()
    plt.plot(index, loss)
    plt.ylim(0.0, 0.02)
    plt.title("Train Loss")
    
    plt.figure()
    plt.plot(index, acc)  
    plt.ylim(0.95, 1.0)
    plt.title("Train Accuracy")


    plt.figure()
    plt.plot(index, dice)
    plt.title("Train Dice Coefficient")

    plt.figure()
    plt.plot(valid_idx, valid_acc)
    plt.title("Test Accuracy")

    plt.figure()
    plt.plot(valid_idx, valid_dice)
    plt.title("Test Dice Coefficient")

    plt.show()
plot_log('log/Unet_log_5.txt')