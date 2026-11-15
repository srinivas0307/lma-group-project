"""Actual-parameter precision checks shared with the strict collector; no torch import on CPU."""
def validate_precision_record(role,precision):
    if role not in ['generator','agent']:raise ValueError('Unexpected model role')
    expected='nf4' if role=='agent' else 'none'
    if precision.get('quantization')!=expected or precision.get('compute_dtype')!='torch.float16' or precision.get('floating_parameter_dtypes')!=['torch.float16']:
        raise ValueError(f'{role} requires {expected} weights with float16 computation')
    if role=='agent' and (precision.get('quantized_parameter_count',0)<=0 or precision.get('actual_quant_types')!=['nf4']):
        raise ValueError('Actual NF4 repair parameters not verified')

def inspect_precision(role,model):
    parameters=list(model.parameters())
    quantized=[p for p in parameters if type(p).__name__=='Params4bit']
    floating=sorted({str(p.dtype) for p in parameters if p.is_floating_point()})
    types=sorted({str(getattr(p,'quant_type',None)) for p in quantized})
    if role=='generator' and (quantized or getattr(model,'is_loaded_in_4bit',False) or getattr(model,'is_loaded_in_8bit',False)):
        raise ValueError('Generator must be unquantized FP16')
    if role=='agent' and getattr(model,'is_loaded_in_8bit',False):raise ValueError('Repair must use NF4, not 8-bit')
    compute=sorted({str(m.compute_dtype) for m in model.modules() if type(m).__name__=='Linear4bit'})
    if role=='agent' and compute!=['torch.float16']:raise ValueError('NF4 linear computation must use float16')
    precision=dict(quantization='nf4' if quantized else 'none',compute_dtype='torch.float16',floating_parameter_dtypes=floating,
                   quantized_parameter_count=len(quantized),actual_quant_types=types)
    validate_precision_record(role,precision)
    return precision
