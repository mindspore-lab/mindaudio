try:
    import moxing as mox
except:
    pass


def move_from_to(src, dst):
    try:
        mox.file.copy_parallel(src, dst)
        print("Successfully Upload {} to {}".format(src, dst))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(src, dst) + str(e))
    return
