import math
def extend_anno(row_boxes,col_boxes,imgshape,extend_pix=8,keep_pix=2):  # 宽度扩展到对应像素数
    Thtml = []
    table = {
        'separator_col':[],
        'separator_row':[],
    }
    for row in row_boxes:
        table['separator_row'].append([row[1],row[3]])
    for col in col_boxes:
        table['separator_col'].append([col[0],col[2]])
    Thtml.append(table)
    for table, tid in zip(Thtml, range(0, len(Thtml))):
        # imgshape = table['resize_shape']
        # 计算列并扩展
        scol = table['separator_col']
        # # TODO must be ###
        # scol[0] = [0, 5]
        # scol[1] = [6, 15]
        # scol[2] = [18, 22]
        # scol[3] = [23, 25]
        for scid in range(0, len(table['separator_col'])):
            thickness = scol[scid][1] - scol[scid][0]
            if thickness < extend_pix:
                expix = int((extend_pix - thickness))
                expixhelf = int(math.ceil((extend_pix - thickness) / 2))
                if scid == 0:
                    leftpix = scol[scid][0]
                    # leftpix = 0
                else:
                    leftpix = scol[scid][0] - scol[scid - 1][1]
                    # leftpix = scol[scid-1][1]
                if scid != len(scol) - 1:
                    rightpix = scol[scid + 1][0] - scol[scid][1]
                    # rightpix = scol[scid+1][0]
                else:
                    rightpix = imgshape[1] - scol[scid][1]
                    # rightpix = imgshape[1]
                if leftpix + rightpix < expix + 2 * keep_pix:  # 达不到扩展标准,只扩展到与上下边界邻近1像素
                    Thtml[tid]['separator_col'][scid][0] -= max(leftpix - keep_pix, 0)
                    Thtml[tid]['separator_col'][scid][1] += max(rightpix - keep_pix, 0)
                else:
                    if leftpix <= expixhelf + keep_pix:  # 左边不够，右边够
                        Thtml[tid]['separator_col'][scid][0] -= max(leftpix - keep_pix, 0)
                        Thtml[tid]['separator_col'][scid][1] += expix - leftpix + keep_pix
                    elif rightpix <= expixhelf + keep_pix:  # 右边不够
                        Thtml[tid]['separator_col'][scid][0] -= max(expix - rightpix + keep_pix, 0)
                        Thtml[tid]['separator_col'][scid][1] += max(rightpix - keep_pix, 0)
                    else:
                        Thtml[tid]['separator_col'][scid][0] -= max(expixhelf, 0)
                        Thtml[tid]['separator_col'][scid][1] += max(expixhelf, 0)
                # Thtml[tid]['separator_col'][scid][1] = min(Thtml[tid]['separator_col'][scid][1],imgshape[1])
                #
                # tems = max(scol[scid][0]-expixhelf,0)
                # Thtml[tid]['separator_col'][scid][0] = max(tems,leftpix-1)
                # tems = min(scol[scid][1]+expixhelf,imgshape[1])
                # Thtml[tid]['separator_col'][scid][1] = min(tems, rightpix - 1)
        # 计算行并扩展
        srow = table['separator_row']
        # TODO must be ###
        # srow[0] = [0, 5]
        # srow[1] = [6, 15]
        # srow[2] = [18, 22]
        # srow[3] = [23, 25]
        for srid in range(0, len(table['separator_row'])):
            thickness = srow[srid][1] - srow[srid][0]
            if thickness < extend_pix:
                expix = int((extend_pix - thickness))
                expixhelf = int(math.ceil((extend_pix - thickness) / 2))
                if srid == 0:
                    leftpix = srow[srid][0]
                    # leftpix = 0
                else:
                    leftpix = srow[srid][0] - srow[srid - 1][1]
                    # leftpix = srow[srid - 1][1]
                if srid != len(srow) - 1:
                    rightpix = srow[srid + 1][0] - srow[srid][1]
                    # rightpix = srow[srid + 1][0]
                else:
                    rightpix = imgshape[0] - srow[srid][1]
                    # rightpix = imgshape[0]

                if leftpix + rightpix < expix + keep_pix * 2:  # 达不到扩展标准,只扩展到与上下边界邻近1像素
                    Thtml[tid]['separator_row'][srid][0] -= max(leftpix - keep_pix, 0)
                    Thtml[tid]['separator_row'][srid][1] += max(rightpix - keep_pix, 0)
                else:
                    if leftpix <= expixhelf + keep_pix:  # 左边不够，右边够
                        Thtml[tid]['separator_row'][srid][0] -= max(leftpix - keep_pix, 0)
                        Thtml[tid]['separator_row'][srid][1] += max(expix - leftpix + keep_pix, 0)
                    elif rightpix <= expixhelf + keep_pix:  # 右边不够
                        Thtml[tid]['separator_row'][srid][0] -= max(expix - rightpix + keep_pix, 0)
                        Thtml[tid]['separator_row'][srid][1] += max(rightpix - keep_pix, 0)
                    else:
                        Thtml[tid]['separator_row'][srid][0] -= max(expixhelf, 0)
                        Thtml[tid]['separator_row'][srid][1] += max(expixhelf, 0)
    return Thtml[0]