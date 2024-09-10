import cProfile  
import pstats  
import io 
def profile_code(code_to_run):  
    '''
    创建一个cProfile分析器用于检测各部分代码运行时间
    code_to_run: 要运行的函数
    '''
    pr = cProfile.Profile()  
    pr.enable()  # 启用分析器  
    
    # 运行要分析的代码  
    code_to_run()  
    
    pr.disable()  # 禁用分析器  
    
    # 使用pstats处理输出并按照累计时间进行排序  
    s = io.StringIO()  
    sortby = 'cumulative'  # 可以使用 'tottime' 来按每个函数的执行时间排序  
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)  
    ps.print_stats(10)  # 可以调整数字以显示不同数量的前n个函数  

    # 打印分析结果  
    print(s.getvalue())  