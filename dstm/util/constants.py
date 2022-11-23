import colorsys

def cm_to_inch(cm):
    return cm/2.54

class Constants:
    # Should it rather be depending on d (the number of outputs???)
    # This would then be different pr. dataset. 
    zero_prob_add = 1e-3

    text_width = cm_to_inch(17.2)
    text_height = cm_to_inch(24.3)
    column_sep_ = cm_to_inch(.5)
    column_width = (text_width - column_sep_)/2
    dpi = 1000
    matplotlib_rcparams = {
        "text.usetex": True,
        "font.family": "Helvetica",
        "savefig.dpi": 1000,
        #"font.sans-serif": ["Helvetica"]
    }

    _linestyle_densely_dashed = (0, (5, 1))
    #linestyle_densely_dashed_dotted = (0, (5, 1, 1, 1, 1, 1))
    #_linestyle_densely_dashed_dotted = (0, (15, 1))
    #path_effects_border_line =  (pe.Stroke(linewidth=2, foreground='black'), pe.Normal())
    styles = {
        "ccstm-elu-rev": {"name": "CCSTM-512 (stm)","linestyle": {"color": colorsys.hls_to_rgb(0, .3,  1,), "linestyle":"solid"}, "pointstyle": {"marker": r"$0$", "color": colorsys.hls_to_rgb(0, .3,  1,)}},
        "ccstm-elu-32-rev": {"name": "CCSTM-32 (stm)" , "linestyle": {"color": colorsys.hls_to_rgb(0, .5,  1,), "linestyle": "solid"}, "pointstyle": {"marker": r"$1$", "color": colorsys.hls_to_rgb(0, .5,  1,)}},
        #"dcstm-annealing": {"name": "DCSTM-512 (stm)", "linestyle": {"color": colorsys.hls_to_rgb(0, .7,  1,), "linestyle":"solid"}, "pointstyle": {"marker": r"$2$", "color": colorsys.hls_to_rgb(0, .7,  1,)}},
        "dcstm-rev": {"name": "DCSTM-512 (stm)", "linestyle": {"color": colorsys.hls_to_rgb(0, .7,  1,), "linestyle":"solid"}, "pointstyle": {"marker": r"$2$", "color": colorsys.hls_to_rgb(0, .7,  1,)}},
        #"io-mc-0": {"name": "MC-0 (stm)" , "linestyle": {"color": colorsys.hls_to_rgb(0.1, .4,  1,), "linestyle":"solid"}, "pointstyle": {"marker": r"$3$", "color": colorsys.hls_to_rgb(0.1, .4,  1,)}},
        "io-mc-3": {"name": "MC-3 (stm)", "linestyle": {"color": colorsys.hls_to_rgb(0.1, .6,  1,), "linestyle":"solid"}, "pointstyle": {"marker": r"$3$", "color": colorsys.hls_to_rgb(0.1, .6,  1,)}},
        'ppm': {"name": "PPM (stm)", "linestyle": {"color": colorsys.hls_to_rgb(.7, .5,  1,), "linestyle":"solid"}, "pointstyle": {"marker": r"$4$", "color": colorsys.hls_to_rgb(.7, .5,  1,)}}, 
        'repetition': {"name": "Repetion (stm)", "linestyle": {"color": colorsys.hls_to_rgb(.9, .5,  1,), "linestyle":"solid"},  "pointstyle": {"marker": r"$5$", "color": colorsys.hls_to_rgb(.9, .5,  1,)}},
        'ltm-dccnn-rev': {"name": "WaveNet-512 (ltm)", "linestyle": {"color": colorsys.hls_to_rgb(.4, .35,  1,), "linestyle": _linestyle_densely_dashed}, "pointstyle": {"marker": r"$6$", "color": colorsys.hls_to_rgb(0.4, .35,  1,)}},
        'ltm-transformer-rel': {"name":"Transformer-512 (ltm)", "linestyle": {"color": colorsys.hls_to_rgb(.6, .3,  1,), "linestyle": _linestyle_densely_dashed}, "pointstyle": {"marker": r"$7$", "color": colorsys.hls_to_rgb(0.6, .3,  1,)}},
        'ltm-transformer-rel-32-rev': {"name": "Transformer-32 (ltm)", "linestyle": {"color": colorsys.hls_to_rgb(.6, .5,  1,), "linestyle": _linestyle_densely_dashed}, "pointstyle": {"marker": r"$8$", "color": colorsys.hls_to_rgb(0.6, .5,  1,)}},
        #'ltm-transformer-lin': {"name":"Transformer-lin (ltm)", "linestyle": {"color": colorsys.hls_to_rgb(.6, .7,  1,), "linestyle": _linestyle_densely_dashed}, "pointstyle": {"marker": r"$10$", "color": colorsys.hls_to_rgb(0.6, .7,  1,)}}
    }