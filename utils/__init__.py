



def main_process(args):
    # FIXME: args is not reasonable and not make sense!
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)



