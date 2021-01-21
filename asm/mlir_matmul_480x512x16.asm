b/matmul/matmul_480x512x16:
(__TEXT,__text) section
_rtclock:
0000000100003b68	sub	sp, sp, #0x30
0000000100003b6c	stp	x29, x30, [sp, #0x20]   ; Latency: 6
0000000100003b70	add	x29, sp, #0x20
0000000100003b74	add	x0, sp, #0x8
0000000100003b78	sub	x1, x29, #0x8
0000000100003b7c	bl	0x100003edc ; symbol stub for: _gettimeofday
0000000100003b80	cbz	w0, 0x100003b94
0000000100003b84	str	x0, [sp]                ; Latency: 4
0000000100003b88	adr	x0, #0x3e0 ; literal pool for: "Error return from gettimeofday: %d"
0000000100003b8c	nop
0000000100003b90	bl	0x100003ee8 ; symbol stub for: _printf
0000000100003b94	ldr	d0, [sp, #0x8]          ; Latency: 4
0000000100003b98	scvtf	d0, d0                  ; Latency: 2
0000000100003b9c	ldr	s1, [sp, #0x10]         ; Latency: 4
0000000100003ba0	sshll.2d	v1, v1, #0x0    ; Latency: 2
0000000100003ba4	scvtf	d1, d1                  ; Latency: 2
0000000100003ba8	nop
0000000100003bac	ldr	d2, 0x100003f60         ; Latency: 4
0000000100003bb0	fmul	d1, d1, d2              ; Latency: 5
0000000100003bb4	fadd	d0, d1, d0              ; Latency: 5
0000000100003bb8	ldp	x29, x30, [sp, #0x20]   ; Latency: 4
0000000100003bbc	add	sp, sp, #0x30
0000000100003bc0	ret
_init_matrix:
0000000100003bc4	stp	x26, x25, [sp, #-0x50]! ; Latency: 6
0000000100003bc8	stp	x24, x23, [sp, #0x10]   ; Latency: 6
0000000100003bcc	stp	x22, x21, [sp, #0x20]   ; Latency: 6
0000000100003bd0	stp	x20, x19, [sp, #0x30]   ; Latency: 6
0000000100003bd4	stp	x29, x30, [sp, #0x40]   ; Latency: 6
0000000100003bd8	add	x29, sp, #0x40
0000000100003bdc	cmp	w2, #0x1
0000000100003be0	b.lt	0x100003c38
0000000100003be4	cmp	w1, #0x1
0000000100003be8	b.lt	0x100003c38
0000000100003bec	mov	x19, x0                 ; Latency: 2
0000000100003bf0	mov	x20, #0x0
0000000100003bf4	mov	w21, w1                 ; Latency: 2
0000000100003bf8	mov	w22, w2                 ; Latency: 2
0000000100003bfc	lsl	x23, x21, #2
0000000100003c00	mov	w24, #0x30000000
0000000100003c04	mov	x25, x21                ; Latency: 2
0000000100003c08	mov	x26, x19                ; Latency: 2
0000000100003c0c	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003c10	scvtf	s0, w0                  ; Latency: 10
0000000100003c14	fmov	s1, w24                 ; Latency: 5
0000000100003c18	fmul	s0, s0, s1              ; Latency: 4
0000000100003c1c	str	s0, [x26], #0x4         ; Latency: 4
0000000100003c20	subs	x25, x25, #0x1
0000000100003c24	b.ne	0x100003c0c
0000000100003c28	add	x20, x20, #0x1
0000000100003c2c	add	x19, x19, x23           ; Latency: 2
0000000100003c30	cmp	x20, x22                ; Latency: 2
0000000100003c34	b.ne	0x100003c04
0000000100003c38	ldp	x29, x30, [sp, #0x40]   ; Latency: 4
0000000100003c3c	ldp	x20, x19, [sp, #0x30]   ; Latency: 4
0000000100003c40	ldp	x22, x21, [sp, #0x20]   ; Latency: 4
0000000100003c44	ldp	x24, x23, [sp, #0x10]   ; Latency: 4
0000000100003c48	ldp	x26, x25, [sp], #0x50   ; Latency: 4
0000000100003c4c	ret
_main:
0000000100003c50	sub	sp, sp, #0x50
0000000100003c54	stp	d9, d8, [sp, #0x20]     ; Latency: 6
0000000100003c58	stp	x20, x19, [sp, #0x30]   ; Latency: 6
0000000100003c5c	stp	x29, x30, [sp, #0x40]   ; Latency: 6
0000000100003c60	add	x29, sp, #0x40
0000000100003c64	add	x0, sp, #0x8
0000000100003c68	add	x1, sp, #0x18
0000000100003c6c	bl	0x100003edc ; symbol stub for: _gettimeofday
0000000100003c70	cbz	w0, 0x100003c84
0000000100003c74	str	x0, [sp]                ; Latency: 4
0000000100003c78	adr	x0, #0x2f0 ; literal pool for: "Error return from gettimeofday: %d"
0000000100003c7c	nop
0000000100003c80	bl	0x100003ee8 ; symbol stub for: _printf
0000000100003c84	ldr	x19, [sp, #0x8]         ; Latency: 4
0000000100003c88	ldr	s0, [sp, #0x10]         ; Latency: 4
0000000100003c8c	sshll.2d	v0, v0, #0x0    ; Latency: 2
0000000100003c90	scvtf	d0, d0                  ; Latency: 2
0000000100003c94	nop
0000000100003c98	ldr	d8, 0x100003f60         ; Latency: 4
0000000100003c9c	fmul	d9, d0, d8              ; Latency: 5
0000000100003ca0	mov	w20, #0x1e0
0000000100003ca4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003ca8	subs	x20, x20, #0x1
0000000100003cac	b.ne	0x100003ca4
0000000100003cb0	mov	w20, #0x1e0
0000000100003cb4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003cb8	subs	x20, x20, #0x1
0000000100003cbc	b.ne	0x100003cb4
0000000100003cc0	mov	w20, #0x1e0
0000000100003cc4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003cc8	subs	x20, x20, #0x1
0000000100003ccc	b.ne	0x100003cc4
0000000100003cd0	mov	w20, #0x1e0
0000000100003cd4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003cd8	subs	x20, x20, #0x1
0000000100003cdc	b.ne	0x100003cd4
0000000100003ce0	mov	w20, #0x1e0
0000000100003ce4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003ce8	subs	x20, x20, #0x1
0000000100003cec	b.ne	0x100003ce4
0000000100003cf0	mov	w20, #0x1e0
0000000100003cf4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003cf8	subs	x20, x20, #0x1
0000000100003cfc	b.ne	0x100003cf4
0000000100003d00	mov	w20, #0x1e0
0000000100003d04	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d08	subs	x20, x20, #0x1
0000000100003d0c	b.ne	0x100003d04
0000000100003d10	mov	w20, #0x1e0
0000000100003d14	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d18	subs	x20, x20, #0x1
0000000100003d1c	b.ne	0x100003d14
0000000100003d20	mov	w20, #0x1e0
0000000100003d24	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d28	subs	x20, x20, #0x1
0000000100003d2c	b.ne	0x100003d24
0000000100003d30	mov	w20, #0x1e0
0000000100003d34	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d38	subs	x20, x20, #0x1
0000000100003d3c	b.ne	0x100003d34
0000000100003d40	mov	w20, #0x1e0
0000000100003d44	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d48	subs	x20, x20, #0x1
0000000100003d4c	b.ne	0x100003d44
0000000100003d50	mov	w20, #0x1e0
0000000100003d54	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d58	subs	x20, x20, #0x1
0000000100003d5c	b.ne	0x100003d54
0000000100003d60	mov	w20, #0x1e0
0000000100003d64	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d68	subs	x20, x20, #0x1
0000000100003d6c	b.ne	0x100003d64
0000000100003d70	mov	w20, #0x1e0
0000000100003d74	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d78	subs	x20, x20, #0x1
0000000100003d7c	b.ne	0x100003d74
0000000100003d80	mov	w20, #0x1e0
0000000100003d84	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d88	subs	x20, x20, #0x1
0000000100003d8c	b.ne	0x100003d84
0000000100003d90	mov	w20, #0x1e0
0000000100003d94	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d98	subs	x20, x20, #0x1
0000000100003d9c	b.ne	0x100003d94
0000000100003da0	scvtf	d0, x19                 ; Latency: 10
0000000100003da4	fadd	d9, d9, d0              ; Latency: 5
0000000100003da8	mov	w19, #0x200
0000000100003dac	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003db0	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003db4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003db8	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dbc	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dc0	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dc4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dc8	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dcc	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dd0	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dd4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dd8	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003ddc	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003de0	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003de4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003de8	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dec	subs	x19, x19, #0x1
0000000100003df0	b.ne	0x100003dac
0000000100003df4	mov	w20, #0x1e0
0000000100003df8	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dfc	subs	x20, x20, #0x1
0000000100003e00	b.ne	0x100003df8
0000000100003e04	add	x19, x19, #0x1
0000000100003e08	cmp	x19, #0x200
0000000100003e0c	b.ne	0x100003df4
0000000100003e10	add	x0, sp, #0x8
0000000100003e14	add	x1, sp, #0x18
0000000100003e18	bl	0x100003edc ; symbol stub for: _gettimeofday
0000000100003e1c	cbz	w0, 0x100003e30
0000000100003e20	str	x0, [sp]                ; Latency: 4
0000000100003e24	adr	x0, #0x144 ; literal pool for: "Error return from gettimeofday: %d"
0000000100003e28	nop
0000000100003e2c	bl	0x100003ee8 ; symbol stub for: _printf
0000000100003e30	ldr	d0, [sp, #0x8]          ; Latency: 4
0000000100003e34	scvtf	d0, d0                  ; Latency: 2
0000000100003e38	ldr	s1, [sp, #0x10]         ; Latency: 4
0000000100003e3c	sshll.2d	v1, v1, #0x0    ; Latency: 2
0000000100003e40	scvtf	d1, d1                  ; Latency: 2
0000000100003e44	fmul	d1, d1, d8              ; Latency: 5
0000000100003e48	fadd	d8, d1, d0              ; Latency: 5
0000000100003e4c	adr	x0, #0x13f ; literal pool for: "FILE_NAME"
0000000100003e50	nop
0000000100003e54	adr	x1, #0x141 ; literal pool for: "w"
0000000100003e58	nop
0000000100003e5c	bl	0x100003ec4 ; symbol stub for: _fopen
0000000100003e60	mov	x19, x0                 ; Latency: 2
0000000100003e64	fsub	d0, d8, d9              ; Latency: 5
0000000100003e68	mov	x8, #0x700000000000
0000000100003e6c	movk	x8, #0x41c7, lsl #48
0000000100003e70	fmov	d1, x8                  ; Latency: 5
0000000100003e74	fdiv	d0, d1, d0              ; Latency: 17
0000000100003e78	mov	x8, #0xcd6500000000
0000000100003e7c	movk	x8, #0x41cd, lsl #48
0000000100003e80	fmov	d1, x8                  ; Latency: 5
0000000100003e84	fdiv	d0, d0, d1              ; Latency: 17
0000000100003e88	str	d0, [sp]                ; Latency: 4
0000000100003e8c	adr	x1, #0x10b ; literal pool for: "%0.2lf GFLOPS\n"
0000000100003e90	nop
0000000100003e94	bl	0x100003ed0 ; symbol stub for: _fprintf
0000000100003e98	mov	x0, x19                 ; Latency: 2
0000000100003e9c	bl	0x100003eb8 ; symbol stub for: _fclose
0000000100003ea0	mov	w0, #0x0
0000000100003ea4	ldp	x29, x30, [sp, #0x40]   ; Latency: 4
0000000100003ea8	ldp	x20, x19, [sp, #0x30]   ; Latency: 4
0000000100003eac	ldp	d9, d8, [sp, #0x20]     ; Latency: 4
0000000100003eb0	add	sp, sp, #0x50
0000000100003eb4	ret
