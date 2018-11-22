function [param] = quicParamParse(paramStr, delimit)
% quicParamParse(paramStr, delimit) - Parses single quic parameter string for value
% QUIC has parameter values in QU_simparams.inp. Each parameter is
% typically a value with an exclamation point after describing the value,
% all on a single line. When reading the file in, the lines are stored as
% cell. This function takes a cell (line) in, and returns the parameter
% value as a number. The delimit input dictates how to split. No delimit
% argument will split on any whitespace

if nargin == 1
    strArr = strsplit(paramStr);
else
    strArr = strsplit(paramStr, delimit);    
end
param = str2double(strArr{1});

end

