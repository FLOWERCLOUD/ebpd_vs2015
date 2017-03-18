#ifndef OBJECT_H
#define OBJECT_H
#pragma once

#include <memory>
#include <vector>
#include <d3d11.h>
#include <DirectXMath.h>
#include <iostream>

struct SSDRObject
{
public:
    //using SharedPtr = std::shared_ptr<Object>;
    //using WeakPtr = std::weak_ptr<Object>;
    //using UniquePtr = std::unique_ptr<Object>;
	typedef std::shared_ptr<SSDRObject> SharedPtr;
	typedef std::weak_ptr<SSDRObject> WeakPtr;
	typedef std::unique_ptr<SSDRObject> UniquePtr;

protected:
    WeakPtr self;
    WeakPtr parent;
    WeakPtr root;
    std::vector<SharedPtr> children;
    bool isVisible;

public:
    static SharedPtr CreateInstance()
    {
        SharedPtr s(new SSDRObject);
        s->self = s;
        return s;
    }
    virtual ~SSDRObject()
    {
    }

public:
    bool IsVisible() const
    {
        return isVisible;
    }
    virtual void Show()
    {
        isVisible = true;
    }
    virtual void Hide()
    {
        isVisible = false;
    }

public:
    int NumChildren() const
    {
        return static_cast<int>(children.size());
    }
    SharedPtr Child(int id)
    {
        return children[id];
    }

public:
    void AddChild(SharedPtr child)
    {
        children.push_back(child);
        child->parent = self;
        child->SetRoot(parent.lock() == nullptr ? self.lock() : root.lock());
    }
    void SetRoot(SharedPtr r)
    {
        if (root.lock() == r)
        {
            return;
        }
        root = r;
        for (auto it = children.begin(); it != children.end(); ++it)
        {
            (*it)->SetRoot(r);
        }
    }

public:
    virtual bool OnInit(ID3D11Device* device, ID3D11DeviceContext* deviceContext, const UINT width, const UINT height)
    {
        for (auto it = children.begin(); it != children.end(); ++it)
        {
            (*it)->OnInit(device, deviceContext, width, height);
        }
        return true;
    }
    virtual void OnDestroy()
    {
        for (auto it = children.begin(); it != children.end(); ++it)
        {
            (*it)->OnDestroy();
        }
        children.clear();
        parent.reset();
        root.reset();
        self.reset();
    }
    virtual void OnRender(ID3D11Device* device, ID3D11DeviceContext* deviceContext)
    {
        for (auto it = children.begin(); it != children.end(); ++it)
        {
            (*it)->OnRender(device, deviceContext);
        }
    }
    virtual void OnResize(ID3D11Device* device, ID3D11DeviceContext* deviceContext, const UINT width, const UINT height)
    {
        for (auto it = children.begin(); it != children.end(); ++it)
        {
            (*it)->OnResize(device, deviceContext, width, height);
        }
    }
    virtual void OnUpdate(ID3D11Device* device, ID3D11DeviceContext* deviceContext, float elapsed)
    {
        for (auto it = children.begin(); it != children.end(); ++it)
        {
            (*it)->OnUpdate(device, deviceContext, elapsed);
        }
    }

protected:
    SSDRObject()
        : isVisible(true)
    {
    }
private:
    SSDRObject(const SSDRObject& src);
    void operator =(const SSDRObject& src);
};

#endif //OBJECT_H
