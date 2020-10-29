from spider_net.ops import commons


def hex_parse(h):
    r = int(h[1:3], 16)
    g = int(h[3:5], 16)
    b = int(h[5:7], 16)
    a = int(h[7:9], 16)
    return r, g, b, a


def color_create():
    hexa = str(hex(50))[2:]
    colors = {
        'Identity': {'hex': '#E53935ff'},
        'Sigmoid': {'hex': '#ab0000ff'},
        'BatchNorm': {'hex': '#000000ff'},
        'ReLU': {'hex': '#7a0000ff'},
        'Avg_Pool_3x3': {'hex': '#5050ffff'},
        'Max_Pool_3x3': {'hex': '#0000ffff'},
        'Max_Pool_5x5': {'hex': '#0000cfff'},
        'Max_Pool_7x7': {'hex': '#00009fff'},
        'Conv_1x1': {'hex': '#fff000ff'},
        'Conv_3x3': {'hex': '#ffd000ff'},
        'Conv_5x5': {'hex': '#ffb000ff'},
        'Conv_7x7': {'hex': '#ff9000ff'},
        'SE_8': {'hex': '#ff61adff'},
        'SE_16': {'hex': '#d44c8eff'},
        'Sep_Conv_3x3': {'hex': '#961696ff'},
        'Sep_Conv_5x5': {'hex': '#761676ff'},
        'Sep_Conv_3x3/8': {'hex': '#ffd000ff'},
        'Sep_Conv_5x5g/8': {'hex': '#ffb000ff'},
        'ISC_3x3': {'hex': '#ff87f5ff'},
        'ISC_5x5': {'hex': '#d100c0ff'},
        'Sep_Conv_7x7': {'hex': '#561656ff'},
        'Dil_Conv_3x3': {'hex': '#009a46ff'},
        'Dil_Conv_5x5': {'hex': '#007a26ff'},
        'Dil_Conv_7x7': {'hex': '#005a06ff'},
        'ContConv': {'hex': '#000000ff'},

    }
    colors = {op: v for op, v in colors.items() if op in commons.keys()}
    colors = {op: {'hex': colors[op]['hex'], 'pos': i} for i, op in enumerate(colors.keys())}

    for color in colors.values():
        color['hexa'] = color['hex'][:-2] + hexa
        color['rgb'] = hex_parse(color['hex'])
        color['rgba'] = hex_parse(color['hexa'])
    return colors