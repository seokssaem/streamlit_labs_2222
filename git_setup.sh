#!/bin/bash
# git_setup.sh — 수업 시작 시 실행

# 이전 자격증명 삭제
cmdkey /delete:LegacyGeneric:target=git:https://github.com 2>/dev/null

# 사용자 정보 입력
read -p "GitHub 사용자명 입력: " username
read -p "GitHub 이메일 입력: " email

git config --global user.name "$username"
git config --global user.email "$email"

echo ""
echo "✅ 설정 완료!"
echo "   이름  : $username"
echo "   이메일: $email"
echo ""
echo "이제 push 할 때 토큰을 입력하세요."