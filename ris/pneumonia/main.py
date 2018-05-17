import dill


def process_report(text):
    
    return "processed:" + text


def pack():
    with open(
            '/Users/giga/Dev/USB/workshop/shipyard/ris/pneumonia/output/model.dill',
            'wb') as f:
        dill.dump(process_report, f)


if __name__ == '__main__':
    print('running')
    pack()
    print('done')

