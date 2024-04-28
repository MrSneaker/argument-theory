
from helpers import *

text = "[r1] ->c\n\
[r2] d->!b\n\
[r3] =>a\n\
[r4] =>b\n\
[r5] a,b=>!c\n\
[r6] c=>d\n\
[r7] d=>!a\n\
[r8] !b=>!r6"

rs = [line2rule(line) for line in text.strip("\n").strip(" ").split("\n")]
print([x.isDefeasible for x in rs])

